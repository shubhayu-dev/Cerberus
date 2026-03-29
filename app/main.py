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

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx

from auth_middleware import _decode_token, extract_roles, require_auth
from agent import detect_failure, generate_remediation_script
from security import scrubber
from vault import vault
from zero_trust import filter_instance, SecurityViolationError
from signing import sign_remediation_payload
from stepup import (
    StepUpContext, check_required, verify_token_has_mfa,
    raise_stepup_required,
)

from upstash_redis import Redis

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

try:
    redis = Redis(
        url=os.getenv("UPSTASH_REDIS_REST_URL"), 
        token=os.getenv("UPSTASH_REDIS_REST_TOKEN")
    )
except Exception as e:
    redis = None
    print(f"SECURITY AUDIT: ⚠ Redis not initialized. Rate limiting disabled. {e}")

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
            "sub":           payload.get("sub"),
            "email":         payload.get("email", "unknown"),
            "name":          payload.get("name", "unknown"),
            "roles":         roles,
            "is_admin":      "admin" in roles,
            "token_payload": payload,
        }

    raise HTTPException(status_code=401, detail="Not authenticated")


def normalize_severity(severity: str) -> str:
    sev = (severity or "").strip().lower()
    if sev not in {"critical", "high", "medium", "low"}:
        return "medium"
    return sev


def adjust_severity_by_risk(severity: str, risk_assessment: int) -> str:
    if severity == "critical" and risk_assessment < 70:
        return "high"
    if severity == "high" and risk_assessment < 40:
        return "medium"
    if severity == "medium" and risk_assessment < 20:
        return "low"
    return severity

async def rate_limiter(request: Request):
    if not redis:
        return # Skip if Redis isn't configured
        
    actor = get_actor(request)
    user_id = actor["sub"]
    
    key = f"rate_limit:{user_id}"
    
    current_count = redis.incr(key)
    if current_count == 1:
        redis.expire(key, 60)

    if current_count > 3:  # Allow 3 requests per minute
        print(f"SECURITY AUDIT: 🚨 Rate limit triggered for user {user_id}")
        raise HTTPException(
            status_code=429, 
            detail="Security Rate Limit: Too many requests. Please wait 1 minute."
        )

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
        #"connection":    "github",
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
        #"connection":    "github",
        "prompt":        "login",   # force re-auth even if session exists
        "state":         "mfa_stepup",  # callback uses this to set mfa_verified=True in session
    })
    print("SECURITY AUDIT: 🔐 Redirecting to Auth0 step-up MFA")
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
        # Check if this is an MFA step-up callback (state=mfa_stepup)
        is_mfa = body.get("state", "") == "mfa_stepup"
        user["mfa_verified"] = is_mfa
        request.session["user"] = user
        print(f"SECURITY AUDIT: Session created | user={user['sub']} | "
              f"role={'admin' if user['is_admin'] else 'user'} | "
              f"mfa_verified={is_mfa}")
        return {"ok": True, "mfa_verified": is_mfa}
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


class IssueRequest(BaseModel):
    repo:  str
    title: str
    body:  str


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

@app.post("/logs/analyze", response_model=RemediationResponse, tags=["Agent"], dependencies=[Depends(rate_limiter)])
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

    raw_severity = remediation.get("severity", "medium")
    raw_risk     = remediation.get("risk_assessment", 50)

    try:
        risk_assessment = int(raw_risk)
    except (TypeError, ValueError):
        risk_assessment = 50

    severity = normalize_severity(raw_severity)
    adjusted_severity = adjust_severity_by_risk(severity, risk_assessment)

    if adjusted_severity != severity:
        print(f"SECURITY AUDIT: Severity adjusted by risk assessment | req={request_id} | "
              f"original={severity} | risk={risk_assessment} -> adjusted={adjusted_severity}")

    severity = adjusted_severity

    print(f"SECURITY AUDIT: AI diagnosis complete | req={request_id} | "
          f"severity={severity} | risk_assessment={risk_assessment} | confidence={remediation.get('confidence')}%")

    # ── Step 5: Permission-based command execution logic ──────────────────────
    signed_payload_dict = None
    security_audit      = {}

    # Determine if command should be executed based on role and severity
    can_execute_command = False
    
    if permission_level == "admin":
        # Admins can execute high/medium/low severity commands directly
        can_execute_command = remediation.get("command") and severity in ["high", "medium", "low"]

        if remediation.get("command") and severity == "critical":
            stepup_ctx = StepUpContext(
                severity=severity,
                failure_category=failure["category"],
                permission_level=permission_level,
                request_id=request_id,
            )
            stepup = check_required(stepup_ctx)

            if stepup.required:
                session_user = get_session_user(request) or {}
                session_mfa = session_user.get("mfa_verified", False)
                token_mfa = verify_token_has_mfa(token_payload, request_id)
                has_mfa = request_body.mfa_verified or session_mfa or token_mfa

                if not has_mfa:
                    print(f"SECURITY AUDIT: 🚨 Step-up MFA required | req={request_id} → raising 403 with stepup flow")
                    raise_stepup_required(stepup)

                print(f"SECURITY AUDIT: ✅ MFA step-up verified | req={request_id} → admin can execute critical command")
                can_execute_command = True

    elif permission_level == "user":
        # Users can execute only high/medium/low severity commands; critical is read-only diagnostics
        can_execute_command = remediation.get("command") and severity != "critical"

    # ── Step 6: AntiHallucinationFilter & RSA-PSS Signing ──────────────────
    if can_execute_command:
        raw_command = remediation.get("command")
        safe_command, filter_audit = filter_instance.validate_or_null(raw_command)

        if not filter_audit.get("passed") and filter_audit.get("original_command_blocked"):
            remediation["command"] = None
            remediation["command_blocked"] = True
            remediation["block_reason"] = filter_audit.get("reason", "failed_security_filter")
            print(f"SECURITY AUDIT: 🚨 Command BLOCKED by AntiHallucinationFilter | "
                  f"req={request_id} | reason={filter_audit.get('reason')}")
            security_audit["filter"] = filter_audit
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
                        "signed": True,
                        "algorithm": "RSA-PSS-SHA256",
                        "key_source": signed.metadata.get("key_source"),
                    }
                    print(f"SECURITY AUDIT: ✅ Command signed | req={request_id} | "
                          f"algorithm=RSA-PSS-SHA256 | role={permission_level}")
                except Exception as e:
                    print(f"SECURITY AUDIT: ⚠ Signing failed | req={request_id} | error={e}")
                    security_audit["signing"] = {"signed": False, "error": str(e)}
    else:
        # Command not allowed - nullify it
        remediation["command"] = None
        reason = (
            "critical_severity_requires_mfa" if severity == "critical" and permission_level == "admin"
            else "critical_severity_user_role" if severity == "critical" and permission_level == "user"
            else f"{permission_level}_role_restrictions"
        )
        security_audit["filter"] = {"skipped": True, "reason": reason}
        print(f"SECURITY AUDIT: Command execution denied | req={request_id} | "
              f"role={permission_level} | severity={severity} | reason={reason}")

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


@app.post('/api/create-issue', tags=['GitHub'])
async def create_github_issue(request_body: IssueRequest, actor=Depends(require_auth)):
    """Creates a GitHub issue via Auth0 vaulted GitHub token (admin-only)."""

    if 'admin' not in actor.get('roles', []):
        raise HTTPException(status_code=403, detail='Only Admins can auto-generate GitHub issues.')

    github_token = await vault.get_github_token(actor['sub'])
    if not github_token:
        github_token = os.getenv('GITHUB_DEMO_TOKEN')
        if github_token:
            print('SECURITY AUDIT: Using fallback GITHUB_DEMO_TOKEN for GitHub issue creation')

    if not github_token:
        raise HTTPException(status_code=500, detail='Could not retrieve GitHub token from Auth0 Vault or environment variable.')

    github_url = f'https://api.github.com/repos/{request_body.repo}/issues'
    headers = {
        'Authorization': f'token {github_token}',
        'Accept': 'application/vnd.github.v3+json',
        'X-GitHub-Api-Version': '2022-11-28',
    }
    payload = { 'title': request_body.title, 'body': request_body.body }

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(github_url, json=payload, headers=headers)
            if response.status_code == 201:
                data = response.json()
                return { 'status': 'success', 'issue_url': data['html_url'], 'issue_number': data.get('number') }
            raise HTTPException(status_code=response.status_code, detail=response.text)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

# Cache buster commit to force CI/CD rebuild
