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
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

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
        
        # ── THE MAGIC BULLET: STRICT SCHEMA ENFORCEMENT ──
        remediation_schema = {
            "type": "OBJECT",
            "properties": {
                "issue": {"type": "STRING"},
                "service": {"type": "STRING"},
                "root_cause": {"type": "STRING"},
                "reasoning": {"type": "STRING"},
                "confidence": {"type": "INTEGER"},
                "severity": {"type": "STRING"},
                "requires_mfa": {"type": "BOOLEAN"},
                "security_verdict": {"type": "STRING"},
                "blast_radius": {"type": "STRING"},
                "risk_assessment": {"type": "INTEGER"},
                "command": {"type": "STRING", "nullable": True},
                "safe_alternatives": {"type": "ARRAY", "items": {"type": "STRING"}},
                "suggested_fix": {"type": "STRING"},
                "rollback": {"type": "STRING"},
                "estimated_downtime": {"type": "STRING"}
            },
            "required": [
                "issue", "service", "root_cause", "reasoning", "confidence", 
                "severity", "requires_mfa", "security_verdict", "blast_radius", 
                "risk_assessment", "command", "safe_alternatives", 
                "suggested_fix", "rollback", "estimated_downtime"
            ]
        }

        _vertex_model = GenerativeModel(
            GEMINI_MODEL,
            generation_config=GenerationConfig(
                temperature=0.2,
                max_output_tokens=8192,
                response_mime_type="application/json",
                response_schema=remediation_schema, # <-- Forces perfect output
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
    # flag it as unstructured so Vertex AI processes it.
    if any(kw in lower for kw in ERROR_KEYWORDS):
        return {
            "detected": True,
            "category": "unstructured_traceback",
            "pattern":  "AI_ANALYSIS_REQUIRED",
            "evidence": "Raw application exception detected",
            "exit_code": _extract_exit_code(lower),
        }

    return {"detected": False, "category": None}


def _extract_exit_code(log_lower: str) -> int | None:
    match = re.search(r"exit(?:ed)?\s+(?:with\s+)?(?:code\s+)?(\d+)", log_lower)
    return int(match.group(1)) if match else None


def _get_safe_alternatives_for_category(category: str) -> list:
    """
    Returns context-specific diagnostic commands based on failure category.
    Each command is safe, read-only, and tailored to investigate the root cause.
    """
    alternatives = {
        "port_conflict": [
            "lsof -i :PORT | head -20",  # Find process using the port
            "netstat -tuln | grep LISTEN",  # List all listening ports
            "sudo systemctl status | grep failed",  # Check for failed services
        ],
        "disk_full": [
            "df -h",  # Disk usage summary
            "du -sh /* | sort -h | tail -10",  # Largest directories
            "find / -type f -size +100M 2>/dev/null | head -20",  # Find large files
        ],
        "permission_denied": [
            "ls -la /application",  # Check file permissions
            "id",  # Current user and groups
            "sudo -l",  # Check sudo privileges
        ],
        "oom_kill": [
            "free -h",  # Memory usage
            "ps aux --sort=-%mem | head -15",  # Top memory consumers
            "dmesg | grep -i oom | tail -5",  # OOM killer logs
        ],
        "service_crash": [
            "systemctl status application",  # Service status
            "journalctl -u application -n 50 --no-pager",  # Service logs
            "ps aux | grep application",  # Check if running
        ],
        "connection_refused": [
            "nc -zv HOST PORT",  # Test connection
            "netstat -tuln | grep ESTABLISHED",  # Active connections
            "curl -v http://HOST:PORT/health 2>&1 | head -20",  # Health check
        ],
        "ssl_error": [
            "openssl s_client -connect HOST:PORT -showcerts",  # Certificate info
            "openssl x509 -in /path/to/cert.pem -noout -dates",  # Certificate expiry
            "update-ca-certificates --verbose",  # Verify CA bundle
        ],
        "timeout": [
            "ping -c 3 HOST",  # Network connectivity
            "timeout 5 curl -v http://HOST:PORT",  # Timeout test
            "netstat -an | grep -i time_wait | wc -l",  # Check TCP state
        ],
        "unstructured_traceback": [
            "tail -100 /var/log/application.log",  # Recent logs
            "journalctl -n 100 --no-pager",  # System journal
            "ps aux | grep -E 'python|node|java'",  # Check running processes
        ],
    }
    
    # Return category-specific commands ONLY, no generic fallback
    return alternatives.get(category, [])


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Cerberus, a Zero-Trust Autonomous IT Remediation Agent.
Your mission is to diagnose system failures and generate precise remediations while enforcing a strict Human-in-the-Loop security protocol.

═══ IDENTITY-BASED PERMISSION RULES (NON-NEGOTIABLE) ═══

permission_level = "user" (Junior Dev):
  - Can execute commands if risk_assessment < 50
  - CANNOT execute if risk_assessment >= 50 (no MFA, just denial)
  - NEVER allowed MFA - users cannot escalate privilege
  - Default to non-destructive commands: truncate/echo "" over rm, systemctl restart over kill -9

permission_level = "admin" (Senior Dev):
  - Can execute commands if risk_assessment < 80
  - Requires MFA if risk_assessment >= 80 (must verify MFA before executing high-risk commands)
  - Provide specific rollback procedure for any action
  - Explain the blast_radius: what data/services are affected, is there a backup?

═══ ANTI-RAMPAGE PROTOCOL ═══
- Always prefer the LEAST destructive command that solves the problem
- If you suggest a restart, explain which dependency failed and the ripple effect
- Prefer truncate over rm for log files
- Prefer systemctl restart over kill -9
- Never suggest commands that affect /etc, /boot, /sys, /proc

═══ ROLLBACK PROCEDURE (CRITICAL REQUIREMENT) ═══
EVERY command MUST have a specific, actionable rollback procedure.
  - File deletions: show restore/recovery command
  - Service restarts: show reverse systemctl command
  - Config changes: show config file restore or git revert
  - NEVER use generic text: always provide exact, copy-paste-ready commands

═══ MANDATORY OUTPUT SCHEMA ═══
Return ONLY a raw, perfectly formatted JSON object. 
DO NOT wrap it in ```json blocks. DO NOT use trailing commas. 
CRITICAL: You MUST properly escape all double quotes (\\") and newlines (\\n) inside your string values. Never use literal newlines inside a JSON string.

{
  "issue": "<concise title — what broke>",
  "service": "<affected service/component>",
  "root_cause": "<specific technical root cause — not generic>",
  "reasoning": "<step-by-step diagnosis — MUST explicitly state permission level>",
  "confidence": <integer 0-100>,
  "severity": "<critical|high|medium|low>",
  "requires_mfa": <true only if risk_assessment >= 80 and permission=admin, else false>,
  "security_verdict": "<2-3 sentences: WHY this specific command, what is the blast radius, what could go wrong>",
  "blast_radius": "<what data/services are affected if this command runs — be specific>",
  "risk_assessment": <integer 1-100 — potential for data loss or downtime>,
  "command": "<bash command, null only for user+critical severity>",
  "safe_alternatives": ["<diagnostic cmd 1 — context-specific or empty array if not applicable>", "<diagnostic cmd 2>", "<diagnostic cmd 3>"],
  "suggested_fix": "<human-readable step-by-step fix>",
  "rollback": "<CRITICAL: exact rollback command/procedure — must be specific, not generic>",
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

{"IMPORTANT: This is a USER request. Provide commands if risk_assessment < 50. Deny commands and provide readonly diagnostics if risk_assessment >= 50. NEVER set requires_mfa=true for users." if permission_level == "user" else "IMPORTANT: This is an ADMIN request. Provide the most targeted fix. Set requires_mfa=true ONLY if risk_assessment >= 80."}

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

        raw_text = response.text.strip()

        # ── ROBUST JSON EXTRACTION ──────────────────────────────────────────
        result = _extract_json(raw_text)

        if result is None:
            # Primary extraction failed — run the formatter agent
            log.warning(f"Primary JSON extraction failed — running formatter agent")
            result = await _format_with_agent(raw_text, permission_level)

        if result is None:
            return _api_error_fallback(f"Both extraction and formatter failed. Raw: {raw_text[:200]}")

        # ── SECURITY ENFORCEMENT & SCHEMA VALIDATION ─────────
        
        # 1. Kill 'undefined' frontend bugs by setting safe fallbacks for missing keys
        default_schema = {
            "issue": "Unknown Issue", "service": "unknown", "root_cause": "Unknown",
            "reasoning": "AI did not provide reasoning.", "confidence": 0,
            "severity": "medium", "requires_mfa": False, "security_verdict": "Unknown",
            "blast_radius": "Unknown", "risk_assessment": 50, "command": None,
            "safe_alternatives": ["df -h", "ps aux", "journalctl -xe | tail -50"],
            "suggested_fix": "Review logs manually.", "rollback": "N/A", "estimated_downtime": "Unknown"
        }
        
        # Merge AI result with defaults (AI values overwrite defaults)
        for key, default_val in default_schema.items():
            if key not in result or result.get(key) is None:
                result[key] = default_val

        # 2. Hard enforce: never return executable commands to non-admin roles
        if permission_level != "admin":
            result["command"] = None

        # 3. Hard enforce MFA Logic: NEVER trust the AI to do this correctly
        is_critical = str(result.get("severity", "")).lower() == "critical"
        has_command = bool(result.get("command"))
        
        # Only trigger MFA if it's an admin, it's actually critical, AND there's a command to run
        if is_critical and permission_level == "admin" and has_command:
            result["requires_mfa"] = True
        else:
            result["requires_mfa"] = False

        log.info(f"Vertex AI success | model={GEMINI_MODEL} | confidence={result.get('confidence')}%")
        return result

    except Exception as e:
        log.error(f"Vertex AI call failed: {e}")
        return _api_error_fallback(f"LLM Generation Failed: {str(e)}")

def _extract_json(raw_text: str) -> dict | None:
    """
    Attempts to extract a JSON object from raw LLM output.
    Handles three common Gemini output formats:
      1. Clean JSON (ideal)
      2. ```json ... ``` fenced block
      3. JSON embedded somewhere in a prose response
    Returns parsed dict or None if all attempts fail.
    """
    if not raw_text:
        return None

    # Strategy 1: Direct parse — Gemini returned clean JSON
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Strip markdown fences (```json ... ``` or ``` ... ```)
    # Find the outermost { } pair regardless of surrounding text
    start = raw_text.find('{')
    end   = raw_text.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = raw_text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Strategy 3: Fix common Gemini quirks
    # - Trailing commas before } or ]
    # - Unescaped newlines inside strings
    if start != -1 and end != -1:
        candidate = raw_text[start:end + 1]
        import re as _re
        # Remove trailing commas
        candidate = _re.sub(r',\s*([}\]])', r'\1', candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Strategy 4: Handle TRUNCATED JSON (has { but missing closing })
    # This happens when API response is cut off mid-transmission
    if start != -1 and (end == -1 or end < start):
        # JSON started but wasn't closed properly
        candidate = raw_text[start:]
        
        # Count braces to see how many we need to close
        open_braces = candidate.count('{') - candidate.count('}')
        close_brackets = candidate.count('[') - candidate.count(']')
        
        # Try to complete the JSON by adding missing braces
        repairs = [
            candidate + '}' * open_braces + ']' * close_brackets,  # Add missing closes
            candidate + '"}' * max(1, open_braces),  # Close any open string + braces
            candidate.rstrip(',') + '}' * max(1, open_braces),  # Remove trailing comma + close
        ]
        
        for repaired in repairs:
            try:
                result = json.loads(repaired)
                log.info(f"_extract_json: recovered truncated JSON | added_closes={open_braces}")
                return result
            except json.JSONDecodeError:
                continue

    log.warning(f"_extract_json: all strategies failed | raw_start={raw_text[:80]} | has_open_brace={start != -1}")
    return None


FORMATTER_PROMPT = """You are a JSON formatter. You will receive unstructured text from an AI diagnostic agent.
Your ONLY job is to extract the information and return it as a perfectly valid JSON object.

CRITICAL INSTRUCTIONS:
1. Generate context-specific safe_alternatives ONLY if failure category is clear
   - port_conflict → lsof, netstat, systemctl commands
   - disk_full → df, du, find commands  
   - oom_kill → free, ps (memory), dmesg commands
   - permission_denied → ls -la, id, sudo -l commands
   - service_crash → systemctl status, journalctl, ps commands
   - If category is unclear/generic → use EMPTY ARRAY []

2. Rollback MUST be specific and actionable (CRITICAL requirement)
   - For file deletions: show restore/undo command
   - For service changes: show systemctl rollback command
   - For config changes: show config restore procedure
   - NEVER output generic text like "Unknown" for rollback

3. Command rules based on severity:
   - User role + critical severity → command MUST be null (readonly diagnostics only)
   - User role + high severity → provide actionable remediation command
   - Admin role + critical → provide command with requires_mfa=true
   - Always prefer non-destructive: truncate/echo "" over rm, systemctl restart over kill -9

Return ONLY raw JSON. No markdown. No explanation. No trailing commas.

Required schema:
{
  "issue": "<string>",
  "service": "<string>",
  "root_cause": "<string>",
  "reasoning": "<string>",
  "confidence": <integer 0-100>,
  "severity": "<critical|high|medium|low>",
  "requires_mfa": <boolean>,
  "security_verdict": "<string>",
  "blast_radius": "<string>",
  "risk_assessment": <integer 1-100>,
  "command": <string or null>,
  "safe_alternatives": [<empty if category unclear, else 3 context-specific commands>],
  "suggested_fix": "<string>",
  "rollback": "<string — MUST be specific procedure, not generic>",
  "estimated_downtime": "<string>"
}

If a field cannot be determined:
- strings: "Unknown" (EXCEPT rollback: always provide best-effort specific procedure)
- integers: 50
- booleans: false
- command: null for critical user, specific for high/medium/low user, specific for all admin
- safe_alternatives: [] if unclear, specific commands if clear
- rollback: always specific (e.g., "sudo systemctl restart nginx", "git revert <commit>", "restore /etc/config.bak")"""


async def _format_with_agent(raw_text: str, permission_level: str) -> dict | None:
    """
    Two-agent pattern: when the primary Gemini call returns malformed output,
    this function calls Gemini again with a minimal formatter prompt.
    """
    if not GCP_PROJECT_ID:
        return None

    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel, GenerationConfig

        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        
        # Same strict schema for the formatter
        formatter_schema = {
            "type": "OBJECT",
            "properties": {
                "issue": {"type": "STRING"}, "service": {"type": "STRING"},
                "root_cause": {"type": "STRING"}, "reasoning": {"type": "STRING"},
                "confidence": {"type": "INTEGER"}, "severity": {"type": "STRING"},
                "requires_mfa": {"type": "BOOLEAN"}, "security_verdict": {"type": "STRING"},
                "blast_radius": {"type": "STRING"}, "risk_assessment": {"type": "INTEGER"},
                "command": {"type": "STRING", "nullable": True},
                "safe_alternatives": {"type": "ARRAY", "items": {"type": "STRING"}},
                "suggested_fix": {"type": "STRING"}, "rollback": {"type": "STRING"},
                "estimated_downtime": {"type": "STRING"}
            },
            "required": ["issue", "service", "root_cause", "reasoning", "confidence", "severity", "requires_mfa", "security_verdict", "blast_radius", "risk_assessment", "command", "safe_alternatives", "suggested_fix", "rollback", "estimated_downtime"]
        }

        formatter = GenerativeModel(
            GEMINI_MODEL,
            generation_config=GenerationConfig(
                temperature=0.0,
                max_output_tokens=1024,
                response_mime_type="application/json",
                response_schema=formatter_schema, # <-- Prevents formatter laziness
            ),
        )

        formatter_input = f"""{FORMATTER_PROMPT}\n\nUNSTRUCTURED DIAGNOSTIC TEXT TO FORMAT:\n---\n{raw_text[:3000]}\n---\n\nReturn ONLY the JSON object."""

        log.info("Formatter agent running — attempting to rescue malformed output")
        
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, lambda: formatter.generate_content(formatter_input))
        result = _extract_json(resp.text.strip())

        if result:
            if permission_level != "admin":
                result["command"] = None
            return result
            
        return None

    except Exception as e:
        log.error(f"Formatter agent failed: {e}")
        return None


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
        "security_verdict": "Manual intervention required. AI disabled. Check API logs and GCP quotas.",
        "blast_radius": "Remediation pipeline blocked - manual fixes required",
        "risk_assessment": 100,
        "command": None,
        "safe_alternatives": [],  # No generic commands - category not identified
        "suggested_fix": "1. Check if Vertex AI API is enabled in Cloud Console\n2. Verify service quotas\n3. Review API logs for errors",
        "rollback": "Restart Cerberus service: systemctl restart cerberus-remediation",
        "estimated_downtime": "Until Vertex AI API is restored"
    }