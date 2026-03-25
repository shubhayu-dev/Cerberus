"""
Cerberus — Secure Autonomous IT Remediation Agent
Zero-Trust FastAPI backend + Web Dashboard

Security layers (in order):
1. security.py      → PII scrubbing (emails, phones, API keys)
2. auth_middleware  → Auth0 JWT validation + role extraction
3. zero_trust.py    → AntiHallucinationFilter (syscall whitelist)
4. stepup.py        → Step-up MFA for critical severity incidents
5. signing.py       → RSA-PSS payload signing before command delivery
6. vault.py         → Token Vault GitHub issue creation
"""

import os
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from app.auth_middleware import _decode_token, extract_roles
from app.agent import detect_failure, generate_remediation_script
from app.security import scrubber
from app.vault import vault
from app.zero_trust import filter_instance, SecurityViolationError
from app.signing import sign_remediation_payload
from app.stepup import (
    StepUpContext, check_required, verify_token_has_mfa,
    raise_stepup_required,
)

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("cerberus")

AUTH0_DOMAIN    = os.getenv("AUTH0_DOMAIN")
AUTH0_CLIENT_ID = os.getenv("AUTH0_CLIENT_ID")
AUTH0_AUDIENCE  = os.getenv("AUTH0_AUDIENCE")
AUTH0_CALLBACK  = os.getenv("AUTH0_CALLBACK_URL", "http://localhost:8000/callback")
CLOUD_RUN_URL   = os.getenv("CLOUD_RUN_URL", "https://remediation-agent-gzuqcqtiqa-uc.a.run.app")

TEMPLATES = Path(__file__).parent / "templates"

app = FastAPI(
    title="Cerberus — Secure Autonomous IT Remediation Agent",
    description=(
        "Zero-trust AI agent: Auth0 RBAC + AntiHallucination filter + "
        "RSA-PSS signing + Step-up MFA + Token Vault"
    ),
    version="2.0.0",
)

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("APP_SECRET_KEY", "dev-secret-change-in-prod"),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_template(name: str) -> str:
    return (TEMPLATES / name).read_text()


def get_session_user(request: Request) -> dict | None:
    return request.session.get("user")


def get_actor(request: Request) -> dict:
    """Resolves actor from session or JWT Bearer token."""
    actor = get_session_user(request)
    if actor:
        return actor

    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token   = auth[7:]
        payload = _decode_token(token)
        roles   = extract_roles(payload)
        return {
            "sub":          payload.get("sub"),
            "email":        payload.get("email", "unknown"),
            "name":         payload.get("name", "unknown"),
            "roles":        roles,
            "is_admin":     "admin" in roles,
            "token_payload": payload,
        }

    raise HTTPException(status_code=401, detail="Not authenticated")


# ── Web Routes ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["Web"])
async def landing(request: Request):
    user = get_session_user(request)
    if user:
        return RedirectResponse(url="/dashboard")
    return HTMLResponse(read_template("landing.html"))


@app.get("/login", tags=["Web"])
async def login():
    """Redirects directly to GitHub login via Auth0."""
    from urllib.parse import urlencode
    params = urlencode({
        "response_type": "token",
        "client_id":     AUTH0_CLIENT_ID,
        "redirect_uri":  AUTH0_CALLBACK,
        "audience":      AUTH0_AUDIENCE,
        "scope":         "openid profile email",
        "connection":    "github",
    })
    return RedirectResponse(url=f"https://{AUTH0_DOMAIN}/authorize?{params}")

@app.get("/login/mfa", tags=["Web"])
async def login_mfa():
    """Redirects to Auth0 explicitly demanding Step-Up MFA."""
    from urllib.parse import urlencode
    params = urlencode({
        "response_type": "token",
        "client_id":     AUTH0_CLIENT_ID,
        "redirect_uri":  AUTH0_CALLBACK,
        "audience":      AUTH0_AUDIENCE,
        "scope":         "openid profile email",
        # THIS is the magic string for the Auth0 backend
        "acr_values":    "http://schemas.openid.net/pape/policies/2007/06/multi-factor"
    })
    return RedirectResponse(url=f"https://{AUTH0_DOMAIN}/authorize?{params}")


@app.get("/callback", response_class=HTMLResponse, tags=["Web"])
async def callback():
    return HTMLResponse(read_template("callback.html"))


@app.post("/auth/store-token", tags=["Web"])
async def store_token(request: Request):
    """Validates token and stores user in session."""
    body  = await request.json()
    token = body.get("access_token")
    if not token:
        raise HTTPException(status_code=400, detail="No token provided")

    try:
        payload = _decode_token(token)
        roles   = extract_roles(payload)
        user    = {
            "sub":           payload.get("sub"),
            "email":         payload.get("email", "unknown"),
            "name":          payload.get("name", "unknown"),
            "roles":         roles,
            "is_admin":      "admin" in roles,
            "token":         token,
            "token_payload": payload,
        }
        request.session["user"] = user
        print(f"SECURITY AUDIT: Session created | user={user['sub']} | "
              f"role={'admin' if user['is_admin'] else 'user'} | "
              f"email={user['email']}")
        return {"ok": True}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))


@app.get("/dashboard", response_class=HTMLResponse, tags=["Web"])
async def dashboard(request: Request):
    user = get_session_user(request)
    if not user:
        return RedirectResponse(url="/login")
    return HTMLResponse(read_template("dashboard.html"))


@app.get("/logout", tags=["Web"])
async def logout(request: Request):
    from urllib.parse import urlencode
    request.session.clear()
    params = urlencode({
        "returnTo":  CLOUD_RUN_URL,
        "client_id": AUTH0_CLIENT_ID,
    })
    return RedirectResponse(url=f"https://{AUTH0_DOMAIN}/v2/logout?{params}")


# ── Health & Auth ─────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"], include_in_schema=False)
@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status":  "operational",
        "service": "Cerberus",
        "version": "2.0.0",
        "docs":    "/docs",
        "security_layers": [
            "pii_scrubbing",
            "auth0_rbac",
            "anti_hallucination_filter",
            "step_up_mfa",
            "rsa_pss_signing",
            "token_vault",
        ],
    }


@app.get("/me", tags=["Auth"])
async def me(request: Request):
    actor = get_actor(request)
    return {
        "sub":      actor["sub"],
        "email":    actor["email"],
        "name":     actor["name"],
        "roles":    actor["roles"],
        "is_admin": actor["is_admin"],
    }


# ── Models ────────────────────────────────────────────────────────────────────

class LogAnalysisRequest(BaseModel):
    log_text:          str
    service_name:      Optional[str] = "unknown"
    environment:       Optional[str] = "production"
    github_repo:       Optional[str] = None
    mfa_verified:      bool          = False  # frontend sets True after step-up


class RemediationResponse(BaseModel):
    timestamp:         str
    request_id:        str
    actor:             dict
    service_name:      str
    environment:       str
    failure_detected:  bool
    failure_category:  Optional[str]
    permission_level:  str
    remediation:       Optional[dict]
    signed_payload:    Optional[dict]   # RSA-PSS signed command
    security_audit:    dict             # zero-trust filter results
    github_issue:      Optional[dict]
    audit_trail:       dict


# ── Core Agent Endpoint ───────────────────────────────────────────────────────

@app.post("/logs/analyze", response_model=RemediationResponse, tags=["Agent"])
async def analyze_logs(request_body: LogAnalysisRequest, request: Request):
    """
    Zero-trust remediation pipeline:

    1. Resolve actor (session or JWT)
    2. Scrub PII from log
    3. Detect failure category
    4. Check step-up MFA requirement (critical severity)
    5. Call Vertex AI for diagnosis
    6. Run AntiHallucinationFilter on generated command
    7. Sign approved command with RSA-PSS
    8. Create GitHub issue via Token Vault
    9. Return structured response with full security audit trail
    """
    request_id = str(uuid.uuid4())[:8]

    # ── Step 1: Resolve actor ─────────────────────────────────────────────────
    actor            = get_actor(request)
    permission_level = "admin" if actor["is_admin"] else "user"
    token_payload    = actor.get("token_payload", {})

    print(f"\nSECURITY AUDIT: ═══ Request {request_id} ═══")
    print(f"SECURITY AUDIT: Actor={actor['sub'][:30]} | "
          f"role={permission_level} | service={request_body.service_name} | "
          f"env={request_body.environment}")

    # ── Step 2: Scrub PII ─────────────────────────────────────────────────────
    clean_log = scrubber.scrub(request_body.log_text)
    print(f"SECURITY AUDIT: PII scrubbing complete | req={request_id}")

    # ── Step 3: Detect failure ────────────────────────────────────────────────
    failure = detect_failure(clean_log)

    if not failure["detected"]:
        print(f"SECURITY AUDIT: No failure detected | req={request_id} → no action")
        return RemediationResponse(
            timestamp=datetime.now(timezone.utc).isoformat(),
            request_id=request_id,
            actor={"sub": actor["sub"], "name": actor["name"], "roles": actor["roles"]},
            service_name=request_body.service_name,
            environment=request_body.environment,
            failure_detected=False, failure_category=None,
            permission_level=permission_level,
            remediation=None, signed_payload=None,
            security_audit={"filter": "skipped", "reason": "no_failure_detected"},
            github_issue=None,
            audit_trail={
                "action":    "no_action_required",
                "reason":    "No failure patterns detected",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    print(f"SECURITY AUDIT: Failure detected | req={request_id} | "
          f"category={failure['category']} | evidence={failure['evidence']}")

    # ── Step 4: Vertex AI remediation ─────────────────────────────────────────
    remediation = await generate_remediation_script(
        log_text=clean_log,
        permission_level=permission_level,
        failure=failure,
    )

    severity = remediation.get("severity", "medium")
    print(f"SECURITY AUDIT: AI diagnosis complete | req={request_id} | "
          f"severity={severity} | confidence={remediation.get('confidence')}%")

    # ── Step 5: Step-up MFA check (critical severity, admin only) ─────────────
    signed_payload_dict = None
    security_audit      = {}

    if permission_level == "admin" and remediation.get("command"):

        stepup_ctx = StepUpContext(
            severity=severity,
            failure_category=failure["category"],
            permission_level=permission_level,
            request_id=request_id,
        )
        stepup = check_required(stepup_ctx)

        if stepup.required:
            # Verify if the current token payload actually contains the MFA claim
            # (Usually found in token_payload.get("amr", []) containing "mfa")
            has_mfa = (
                request_body.mfa_verified
                and verify_token_has_mfa(token_payload, request_id)
            )

            if not has_mfa:
                print(f"SECURITY AUDIT: 🚨 Step-up MFA required but not verified | req={request_id} → triggering frontend redirect")
                # THE FIX: Explicitly send the exact JSON the frontend JS is looking for
                raise HTTPException(
                    status_code=403,
                    detail={"error": "mfa_required", "message": "Step-up authentication required for critical severity."}
                )
            else:
                print(f"SECURITY AUDIT: ✅ MFA step-up verified | req={request_id} → proceeding")

        # ── Step 6: AntiHallucinationFilter ──────────────────────────────────
        raw_command = remediation.get("command")
        safe_command, filter_audit = filter_instance.validate_or_null(raw_command)

        if not filter_audit.get("passed") and filter_audit.get("original_command_blocked"):
            remediation["command"] = None
            remediation["command_blocked"] = True
            remediation["block_reason"]    = filter_audit.get("reason", "failed_security_filter")
            print(f"SECURITY AUDIT: 🚨 Command BLOCKED by AntiHallucinationFilter | "
                  f"req={request_id} | reason={filter_audit.get('reason')}")
        else:
            remediation["command"] = safe_command

        security_audit["filter"] = filter_audit

        # ── Step 7: RSA-PSS Signing ───────────────────────────────────────────
        if safe_command:
            try:
                signed = sign_remediation_payload(
                    bash_command=safe_command,
                    request_id=request_id,
                    actor_sub=actor["sub"],
                    failure_category=failure["category"],
                )
                signed_payload_dict = signed.model_dump()
                security_audit["signing"] = {
                    "signed":    True,
                    "algorithm": "RSA-PSS-SHA256",
                    "key_source": signed.metadata.get("key_source"),
                }
                print(f"SECURITY AUDIT: ✅ Command signed | req={request_id} | "
                      f"algorithm=RSA-PSS-SHA256")
            except Exception as e:
                print(f"SECURITY AUDIT: ⚠ Signing failed | req={request_id} | error={e}")
                security_audit["signing"] = {"signed": False, "error": str(e)}

    elif permission_level == "user":
        # Users never get commands — no filter needed
        remediation["command"] = None
        security_audit["filter"] = {"skipped": True, "reason": "user_role_no_commands"}
        print(f"SECURITY AUDIT: User role → command nullified | req={request_id}")

    # ── Step 8: GitHub issue via Token Vault ──────────────────────────────────
    github_issue = None
    if request_body.github_repo:
        print(f"SECURITY AUDIT: Token Vault — fetching GitHub token | "
              f"req={request_id} | user={actor['sub'][:20]}")
        github_token = await vault.get_github_token(actor["sub"])
        if github_token:
            github_issue = await vault.create_incident_issue(
                github_token=github_token,
                repo=request_body.github_repo,
                remediation=remediation,
                failure=failure,
                service_name=request_body.service_name,
                environment=request_body.environment,
                request_id=request_id,
                actor_name=actor["name"],
                permission_level=permission_level,
            )
            if github_issue:
                print(f"SECURITY AUDIT: ✅ GitHub issue created via Token Vault | "
                      f"req={request_id} | issue=#{github_issue.get('number')} | "
                      f"url={github_issue.get('url')}")
        else:
            print(f"SECURITY AUDIT: ⚠ No GitHub token in Auth0 vault | "
                  f"req={request_id} | user must login via GitHub social connection")

    # ── Step 9: Build audit trail ─────────────────────────────────────────────
    audit = {
        "action":              "remediation_generated",
        "permission_level":    permission_level,
        "command_authorized":  bool(remediation.get("command")),
        "command_signed":      bool(signed_payload_dict),
        "filter_passed":       security_audit.get("filter", {}).get("passed", True),
        "mfa_verified":        request_body.mfa_verified,
        "failure_category":    failure["category"],
        "severity":            severity,
        "github_issue_created": bool(github_issue),
        "actor_sub":           actor["sub"],
        "timestamp":           datetime.now(timezone.utc).isoformat(),
    }

    print(f"SECURITY AUDIT: ═══ Request {request_id} complete ═══ | "
          f"command={'authorized' if audit['command_authorized'] else 'denied'} | "
          f"signed={audit['command_signed']} | "
          f"github_issue={audit['github_issue_created']}\n")

    return RemediationResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        request_id=request_id,
        actor={"sub": actor["sub"], "name": actor["name"], "roles": actor["roles"]},
        service_name=request_body.service_name,
        environment=request_body.environment,
        failure_detected=True,
        failure_category=failure["category"],
        permission_level=permission_level,
        remediation=remediation,
        signed_payload=signed_payload_dict,
        security_audit=security_audit,
        github_issue=github_issue,
        audit_trail=audit,
    )


# ── GitHub Progressive Onboarding ─────────────────────────────────────────────

@app.get("/auth/github-consent-url", tags=["Auth"])
async def github_consent_url(request: Request):
    """
    Returns the Auth0 URL to request GitHub public_repo scope.
    Called by the frontend AFTER the user clicks 'Open GitHub Issue'.
    This is the progressive onboarding flow — GitHub access is never
    requested upfront, only when the user explicitly wants it.
    """
    from urllib.parse import urlencode
    actor = get_actor(request)

    params = urlencode({
        "response_type":  "token",
        "client_id":      AUTH0_CLIENT_ID,
        "redirect_uri":   AUTH0_CALLBACK,
        "audience":       AUTH0_AUDIENCE,
        "scope":          "openid profile email",
        "connection":     "github",
        "connection_scope": "public_repo",  # only request what we need
        "prompt":         "consent",        # always show consent screen
        "login_hint":     actor.get("email", ""),
        "state":          "github_consent",
    })
    url = f"https://{AUTH0_DOMAIN}/authorize?{params}"

    print(f"SECURITY AUDIT: GitHub consent URL generated | "
          f"user={actor['sub'][:20]} | scope=public_repo only")

    return {"consent_url": url, "scope_requested": "public_repo"}


@app.get("/demo/logs", tags=["Demo"])
async def get_demo_logs():
    return {"scenarios": [
        {"id": "nginx_crash", "name": "Nginx Port Conflict", "service_name": "nginx",
         "severity": "high",
         "log_text": "2024-01-15 03:42:17 [emerg] bind() to 0.0.0.0:80 failed (98: Address already in use)\nnginx: exited with code 1"},
        {"id": "disk_full", "name": "Disk Space Exhausted", "service_name": "postgresql",
         "severity": "critical",
         "log_text": "2024-01-15 04:15:32 FATAL: No space left on device\nkernel: ENOSPC\npostgresql.service: exited, code=killed"},
        {"id": "permission_denied", "name": "Permission Denied", "service_name": "app-daemon",
         "severity": "high",
         "log_text": "2024-01-15 05:01:44 ERROR: Failed to open /var/run/app/app.sock: Permission denied\napp-daemon.service: Failed to start"},
    ]}


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)