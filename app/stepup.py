"""
stepup.py — Auth0 Step-Up MFA for Critical Severity Incidents
When a critical failure requires high-privilege remediation,
Cerberus pauses execution and demands MFA verification before proceeding.

Flow:
1. Agent detects critical severity
2. stepup.check_required() evaluates if MFA is needed
3. If yes → FastAPI returns 403 with stepup_required=True + auth_url
4. Frontend redirects user to Auth0 MFA prompt
5. User completes MFA → gets new token with acr=mfa claim
6. Frontend retries the request with the MFA-verified token
7. stepup.verify_token_has_mfa() validates the acr claim
8. Execution proceeds

The 'acr' (Authentication Context Class Reference) claim is the
standard OIDC mechanism for step-up auth. Auth0 sets acr=mfa
when the user has completed MFA within the session.
"""

import os
import logging
from datetime import datetime, timezone
from urllib.parse import urlencode
from typing import Optional
from pydantic import BaseModel
from fastapi import HTTPException

log = logging.getLogger("cerberus.stepup")

AUTH0_DOMAIN    = os.getenv("AUTH0_DOMAIN")
AUTH0_CLIENT_ID = os.getenv("AUTH0_CLIENT_ID")
AUTH0_AUDIENCE  = os.getenv("AUTH0_AUDIENCE")
AUTH0_CALLBACK  = os.getenv("AUTH0_CALLBACK_URL", "http://localhost:8000/callback")

# Risk assessment threshold for step-up MFA
# Only admins with risk >= 80 need MFA
MFA_RISK_THRESHOLD = 80


# ── Models ────────────────────────────────────────────────────────────────────

class StepUpRequirement(BaseModel):
    required:     bool
    reason:       Optional[str]  = None
    auth_url:     Optional[str]  = None
    acr_required: str            = "http://schemas.openid.net/pape/policies/2007/06/multi-factor"
    request_id:   Optional[str]  = None


class StepUpContext(BaseModel):
    risk_assessment:  int  # 0-100 score
    failure_category: str
    permission_level: str
    request_id:       str


# ── Step-Up Logic ─────────────────────────────────────────────────────────────

def check_required(context: StepUpContext) -> StepUpRequirement:
    """
    Evaluates whether the current request requires MFA step-up.

    Rules:
    - Admin + risk_assessment >= 80 → requires MFA
    - User role → NEVER requires MFA
    """
    # Users never get MFA
    if context.permission_level != "admin":
        print(f"SECURITY AUDIT: Step-up check | req={context.request_id} | "
              f"role=user → no step-up required (users never need MFA)")
        return StepUpRequirement(required=False, reason="user_role_no_mfa")

    # Check risk assessment threshold
    needs_stepup = context.risk_assessment >= MFA_RISK_THRESHOLD

    if not needs_stepup:
        print(f"SECURITY AUDIT: Step-up check | req={context.request_id} | "
              f"risk_assessment={context.risk_assessment} | category={context.failure_category} → "
              f"below MFA threshold ({MFA_RISK_THRESHOLD})")
        return StepUpRequirement(required=False, reason="below_risk_threshold")

    # Build Auth0 MFA prompt URL
    auth_url = _build_mfa_auth_url(request_id=context.request_id)

    reason = (
        f"High-risk incident (risk_assessment={context.risk_assessment}) requires MFA verification "
        f"before executable commands can be issued. Category: {context.failure_category}"
    )

    print(f"SECURITY AUDIT: 🔐 STEP-UP MFA REQUIRED | req={context.request_id} | "
          f"risk_assessment={context.risk_assessment} | category={context.failure_category} | "
          f"actor_level=admin | reason={reason}")

    return StepUpRequirement(
        required=True,
        reason=reason,
        auth_url=auth_url,
        request_id=context.request_id,
    )


def verify_token_has_mfa(token_payload: dict, request_id: str) -> bool:
    """
    Checks the JWT payload for the MFA acr claim.
    Auth0 sets this when the user completed MFA during the session.

    Returns True if MFA is verified, False otherwise.
    """
    acr = token_payload.get("acr", "")
    amr = token_payload.get("amr", [])  # Authentication Method References

    # Auth0 MFA indicators
    mfa_verified = (
        "mfa" in acr.lower()
        or "http://schemas.openid.net/pape/policies/2007/06/multi-factor" in acr
        or "mfa" in amr
        or "otp" in amr
        or "sms" in amr
    )

    if mfa_verified:
        print(f"SECURITY AUDIT: ✅ MFA verified | req={request_id} | acr={acr} | amr={amr}")
    else:
        print(f"SECURITY AUDIT: ❌ MFA NOT verified | req={request_id} | acr={acr} | amr={amr}")

    return mfa_verified


def raise_stepup_required(stepup: StepUpRequirement):
    """
    Raises an HTTPException that tells the frontend to redirect to MFA.
    The 403 response body contains everything the frontend needs.
    """
    print(f"SECURITY AUDIT: 🚨 Raising step-up 403 | req={stepup.request_id} | "
          f"auth_url configured → frontend will redirect to Auth0 MFA")

    raise HTTPException(
        status_code=403,
        detail={
            "error":          "step_up_required",
            "error_description": stepup.reason,
            "stepup_required": True,
            "auth_url":       stepup.auth_url,
            "acr_required":   stepup.acr_required,
            "request_id":     stepup.request_id,
            "message":        (
                "This critical incident requires Multi-Factor Authentication. "
                "Please complete MFA to authorize command execution."
            ),
        },
    )


def _build_mfa_auth_url(request_id: str) -> str:
    """
    Builds the Auth0 authorize URL that triggers MFA prompt.
    Uses acr_values to request step-up authentication.
    """
    params = urlencode({
        "response_type": "token",
        "client_id":     AUTH0_CLIENT_ID,
        "redirect_uri":  AUTH0_CALLBACK,
        "audience":      AUTH0_AUDIENCE,
        "scope":         "openid profile email",
        "connection":    "github",
        "acr_values":    "http://schemas.openid.net/pape/policies/2007/06/multi-factor",
        "prompt":        "login",  # force re-authentication
        "state":         f"stepup_{request_id}",  # pass request_id through the flow
    })
    return f"https://{AUTH0_DOMAIN}/authorize?{params}"