"""
Remediation Agent — Intelligence Layer
Uses Vertex AI SDK (Gemini 2.5 Flash) with IAM auth.
No API keys — Cloud Run service account handles auth automatically.
Locally: uses Application Default Credentials (gcloud auth application-default login).
"""

import os
import re
import json
import logging
import asyncio
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("remediation-agent")

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION   = os.getenv("GCP_LOCATION", "us-central1")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

_vertex_model = None

def _get_model():
    """
    Lazy-init Vertex AI model.
    - On Cloud Run: authenticates via the attached service account (no key needed).
    - Locally: uses `gcloud auth application-default login` credentials.
    """
    global _vertex_model
    if _vertex_model is not None:
        return _vertex_model

    if not GCP_PROJECT_ID:
        log.warning("GCP_PROJECT_ID not set — will use mock responses")
        return None

    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel, GenerationConfig

        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        _vertex_model = GenerativeModel(
            GEMINI_MODEL,
            generation_config=GenerationConfig(
                temperature=0.2,
                max_output_tokens=1024,
                response_mime_type="application/json",
            ),
        )
        log.info(f"Vertex AI ready | project={GCP_PROJECT_ID} | model={GEMINI_MODEL}")
        return _vertex_model

    except Exception as e:
        log.warning(f"Vertex AI init failed ({e}) — using mock responses")
        return None


# ── Failure Detection ─────────────────────────────────────────────────────────

FAILURE_PATTERNS = {
    "port_conflict":     [r"bind.*address already in use", r"port.*in use", r"\[emerg\].*bind"],
    "disk_full":         [r"no space left on device", r"disk full", r"enospc"],
    "permission_denied": [r"permission denied", r"access denied", r"eacces", r"operation not permitted"],
    "oom_kill":          [r"out of memory", r"oom.kill", r"killed process"],
    "service_crash":     [r"segmentation fault", r"core dumped", r"exited with code [^0]", r"failed to start"],
    "connection_refused":[r"connection refused", r"econnrefused", r"failed to connect"],
    "ssl_error":         [r"ssl_error", r"certificate.*expired", r"ssl handshake failed"],
    "timeout":           [r"timed out", r"timeout", r"operation timed out"],
}

ERROR_KEYWORDS = ["error", "failed", "failure", "crash", "critical", "fatal",
                  "exception", "panic", "abort", "killed"]


def detect_failure(log_text: str) -> dict:
    lower = log_text.lower()

    # 1. Check for specific, known infrastructure patterns
    for category, patterns in FAILURE_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, lower)
            if match:
                return {
                    "detected":  True,
                    "category":  category,
                    "pattern":   pattern,
                    "evidence":  match.group(0),
                    "exit_code": _extract_exit_code(lower),
                }

    # 2. If it's an unknown error (like a Python traceback), 
    # flag it as a generic_error so it still triggers the AI analysis pipeline,
    # but don't trap it in the mock response loop!
    if any(kw in lower for kw in ERROR_KEYWORDS):
        return {
            "detected": True,
            "category": "unstructured_traceback", # <-- This specific string is key
            "pattern":  "AI_ANALYSIS_REQUIRED",
            "evidence": "Raw application exception detected",
            "exit_code": _extract_exit_code(lower),
        }

    return {"detected": False, "category": None}


def _extract_exit_code(log_lower: str) -> int | None:
    match = re.search(r"exit(?:ed)?\s+(?:with\s+)?(?:code\s+)?(\d+)", log_lower)
    return int(match.group(1)) if match else None


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Cerberus, a Zero-Trust Autonomous IT Remediation Agent.
Your mission is to diagnose system failures and generate precise remediations while enforcing a strict Human-in-the-Loop security protocol.

═══ IDENTITY-BASED PERMISSION RULES (NON-NEGOTIABLE) ═══

permission_level = "user" (Junior Dev):
  - Provide READ-ONLY diagnostic commands ONLY (df -h, ls -la, tail, ps, lsof, journalctl)
  - You are STRICTLY PROHIBITED from suggesting destructive commands (rm, kill, systemctl stop, truncate)
  - command field MUST be null
  - safe_alternatives must contain 3 safe read-only commands

permission_level = "admin" (Senior Dev):
  - You MAY suggest high-impact remediations if they are the only logical solution
  - ALWAYS prefer truncate or echo "" over rm for log files (preserves file descriptors)
  - NEVER use recursive force deletes (rm -rf /) on system directories
  - If severity=critical → set requires_mfa=true (triggers Auth0 Step-Up MFA)
  - Explain the blast_radius: what data/services are affected, is there a backup?

═══ ANTI-RAMPAGE PROTOCOL ═══
- Always prefer the LEAST destructive command that solves the problem
- If you suggest a restart, explain which dependency failed and the ripple effect
- Prefer truncate over rm for log files
- Prefer systemctl restart over kill -9
- Never suggest commands that affect /etc, /boot, /sys, /proc

═══ MANDATORY OUTPUT SCHEMA ═══
Return ONLY valid JSON. No markdown. No extra text.

{
  "issue": "<concise title — what broke>",
  "service": "<affected service/component>",
  "root_cause": "<specific technical root cause — not generic>",
  "reasoning": "<step-by-step diagnosis — MUST explicitly state permission level>",
  "confidence": <integer 0-100>,
  "severity": "<critical|high|medium|low>",
  "requires_mfa": <true if severity=critical and permission=admin, else false>,
  "security_verdict": "<2-3 sentences: WHY this specific command, what is the blast radius, what could go wrong>",
  "blast_radius": "<what data/services are affected if this command runs — be specific>",
  "risk_assessment": <integer 1-100 — potential for data loss or downtime>,
  "command": "<single bash command if admin, else null — prefer truncate/restart over rm/kill>",
  "safe_alternatives": ["<read-only diagnostic cmd 1>", "<read-only diagnostic cmd 2>", "<read-only diagnostic cmd 3>"],
  "suggested_fix": "<human-readable step-by-step fix>",
  "rollback": "<exact rollback command or procedure>",
  "estimated_downtime": "<e.g. 2-5 minutes>"
}"""


def _build_prompt(log_text: str, failure: dict, permission_level: str) -> str:
    role_label = "Senior Dev / Admin" if permission_level == "admin" else "Junior Dev / User"
    return f"""{SYSTEM_PROMPT}

═══ CURRENT REQUEST ═══
SYSTEM LOG:
---
{log_text}
---

DETECTED FAILURE CATEGORY: {failure.get('category', 'unknown')}
EVIDENCE: {failure.get('evidence', 'n/a')}
EXIT CODE: {failure.get('exit_code', 'n/a')}
PERMISSION LEVEL: {permission_level} ({role_label})

{"IMPORTANT: This is a USER request. command MUST be null. Provide read-only diagnostics only." if permission_level == "user" else "IMPORTANT: This is an ADMIN request. Provide the most targeted fix. Set requires_mfa=true if severity=critical."}

Analyze this log and return ONLY the JSON object."""


# ── Core AI Tool ──────────────────────────────────────────────────────────────

async def generate_remediation_script(
    log_text: str,
    permission_level: str,
    failure: dict,
) -> dict:
    """
    Calls Vertex AI Gemini 2.5 Flash with a role-gated prompt.
    All log analysis is now handled natively by the LLM.
    """
    model = _get_model()

    if model is None:
        return _api_error_fallback("Vertex AI model is not initialized. Check GCP credentials.")

    prompt = _build_prompt(log_text, failure, permission_level)

    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(prompt)
        )

        result = json.loads(response.text.strip())

        # Hard enforce: never return executable commands to non-admin roles
        if permission_level != "admin":
            result["command"] = None

        log.info(f"Vertex AI success | model={GEMINI_MODEL} | confidence={result.get('confidence')}%")
        return result

    except Exception as e:
        log.error(f"Vertex AI call failed: {e}")
        return _api_error_fallback(f"LLM Generation Failed: {str(e)}")


def _api_error_fallback(error_msg: str) -> dict:
    """Returns a safe schema-compliant error if the Gemini API goes down."""
    return {
        "issue": "AI Agent Unavailable",
        "service": "cerberus-intelligence",
        "root_cause": error_msg,
        "reasoning": "The Vertex AI API could not process the request. Check GCP quotas or connectivity.",
        "confidence": 0,
        "severity": "high",
        "requires_mfa": False,
        "security_verdict": "Manual intervention required. AI disabled.",
        "blast_radius": "Unknown",
        "risk_assessment": 100,
        "command": None,
        "safe_alternatives": ["gcloud services enable aiplatform.googleapis.com"],
        "suggested_fix": "Verify GCP Vertex AI API is enabled and quota is available in the Cloud Console.",
        "rollback": "N/A",
        "estimated_downtime": "Unknown"
    }