"""
TokenVault — Auth0 Management API + GitHub Issue Creator
Retrieves per-user GitHub OAuth tokens stored inside Auth0 identities.
Uses those tokens to open GitHub Issues on failure detection.

Deduplication: before creating a new issue, searches GitHub for an open
issue with the same service + failure category. If found, adds a comment
instead of creating a duplicate.
"""

import os
import httpx
import logging
import time
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("cerberus")

SEVERITY_LABELS = {
    "critical": "priority: critical",
    "high":     "priority: high",
    "medium":   "priority: medium",
    "low":      "priority: low",
}

CATEGORY_LABELS = {
    "port_conflict":      "type: port-conflict",
    "disk_full":          "type: disk-full",
    "permission_denied":  "type: permission-denied",
    "oom_kill":           "type: oom-kill",
    "service_crash":      "type: service-crash",
    "connection_refused": "type: connection-refused",
    "ssl_error":          "type: ssl-error",
    "timeout":            "type: timeout",
    "generic_error":      "type: generic-error",
}

GH_HEADERS = lambda token: {
    "Authorization":        f"token {token}",
    "Accept":               "application/vnd.github.v3+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


class TokenVault:
    def __init__(self):
        self.domain        = os.getenv("AUTH0_DOMAIN")
        self.client_id     = os.getenv("MGMT_CLIENT_ID")
        self.client_secret = os.getenv("MGMT_CLIENT_SECRET")
        self.audience      = f"https://{self.domain}/api/v2/"
        self._issue_cache  = {}

    # ── Auth0 Management API ──────────────────────────────────────────────────

    async def get_mgmt_token(self) -> str | None:
        url = f"https://{self.domain}/oauth/token"
        payload = {
            "client_id":     self.client_id,
            "client_secret": self.client_secret,
            "audience":      self.audience,
            "grant_type":    "client_credentials",
        }
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.post(url, json=payload)
                r.raise_for_status()
                token = r.json().get("access_token")
                log.info("Auth0 mgmt token acquired")
                return token
        except Exception as e:
            log.error(f"Failed to get Auth0 mgmt token: {e}")
            return None

    async def get_github_token(self, user_id: str) -> str | None:
        """
        Retrieves the GitHub OAuth token for a user from Auth0 identities.
        Auth0 stores this automatically when the user logs in via GitHub.
        """
        mgmt_token = await self.get_mgmt_token()
        if not mgmt_token:
            return None

        url     = f"{self.audience}users/{user_id}"
        headers = {"Authorization": f"Bearer {mgmt_token}"}

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(url, headers=headers)
                r.raise_for_status()
                identities = r.json().get("identities", [])
                for identity in identities:
                    if identity.get("provider") == "github":
                        token = identity.get("access_token")
                        log.info(f"GitHub token retrieved for user {user_id[:20]}...")
                        return token
            log.warning(f"No GitHub identity for user {user_id}")
            return None
        except Exception as e:
            log.error(f"Failed to get GitHub token: {e}")
            return None

    # ── Deduplication ─────────────────────────────────────────────────────────

    async def _find_existing_issue(
        self,
        github_token: str,
        repo: str,
        service_name: str,
        failure_category: str,
    ) -> dict | None:
        """
        Searches GitHub for an open issue with the same service + category.
        Uses a 5-minute in-memory cache to prevent GitHub rate limits.
        """
        # ── 1. CHECK CACHE FIRST ──
        cache_key = f"{repo}:{service_name}:{failure_category}"
        cached = self._issue_cache.get(cache_key)
        
        # If it exists in cache and hasn't expired (300 seconds TTL)
        if cached and cached["expires"] > time.time():
            print(f"VAULT: ⚡ CACHE HIT | Skipping GitHub API | key={cache_key}")
            return cached["data"]

        # ── 2. IF NOT CACHED, CALL GITHUB ──
        # Build the search query — matches the title prefix we set on creation
        query = f'repo:{repo} is:issue is:open "[CERBERUS] {service_name}" "{failure_category}"'
        url   = f"https://api.github.com/search/issues?q={quote(query)}&per_page=1"

        print(f"VAULT: Dedup check | repo={repo} | service={service_name} | category={failure_category}")
        print(f"VAULT: Search query → {query}")

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(url, headers=GH_HEADERS(github_token))
                r.raise_for_status()
                data  = r.json()
                total = data.get("total_count", 0)
                items = data.get("items", [])

                if total > 0 and items:
                    existing = items[0]
                    print(
                        f"VAULT: ♻ Duplicate found | issue=#{existing['number']} | "
                        f"title={existing['title'][:60]} | url={existing['html_url']}"
                    )
                    
                    # ── 3. SAVE TO CACHE ──
                    self._issue_cache[cache_key] = {
                        "data": existing, 
                        "expires": time.time() + 300  # Cache for 5 minutes
                    }
                    return existing

                print(f"VAULT: No duplicate found → creating new issue")
                return None

        except httpx.HTTPStatusError as e:
            log.error(f"GitHub search API error {e.response.status_code}: {e.response.text}")
            return None
        except Exception as e:
            log.error(f"Dedup search failed: {e}")
            return None

    async def _add_comment(
        self,
        github_token: str,
        repo: str,
        issue_number: int,
        remediation: dict,
        request_id: str,
        permission_level: str,
    ) -> dict | None:
        """
        Adds a new analysis comment to an existing issue instead of creating a duplicate.
        """
        severity   = remediation.get("severity", "unknown")
        confidence = remediation.get("confidence", 0)
        command    = remediation.get("command")
        risk       = remediation.get("risk_assessment", "N/A")

        comment_body = f"""### 🔄 Re-analysis — `req:{request_id}`

| Field | Value |
|---|---|
| **Permission Level** | `{permission_level}` |
| **Severity** | `{severity.upper()}` |
| **AI Confidence** | `{confidence}%` |
| **Risk Assessment** | `{risk}/100` |

**Root Cause:** {remediation.get('root_cause', 'N/A')}

**Security Verdict:** {remediation.get('security_verdict', 'N/A')}

**Blast Radius:** {remediation.get('blast_radius', 'N/A')}

{'**Command:**' + chr(10) + f'```bash{chr(10)}{command}{chr(10)}```' if command else '**Command:** Not authorized for this role.'}

**Rollback:** `{remediation.get('rollback', 'N/A')}`

---
*Re-analysis by Cerberus · Vertex AI Gemini 2.5 Flash · {request_id}*"""

        url = f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments"

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.post(
                    url,
                    json={"body": comment_body},
                    headers=GH_HEADERS(github_token),
                )
                r.raise_for_status()
                comment = r.json()
                print(
                    f"VAULT: ✅ Comment added to existing issue #{issue_number} | "
                    f"comment_url={comment.get('html_url')}"
                )
                return comment
        except httpx.HTTPStatusError as e:
            log.error(f"GitHub comment error {e.response.status_code}: {e.response.text}")
            return None
        except Exception as e:
            log.error(f"Failed to add comment: {e}")
            return None

    # ── Main Entry Point ──────────────────────────────────────────────────────

    async def create_incident_issue(
        self,
        github_token: str,
        repo: str,
        remediation: dict,
        failure: dict,
        service_name: str,
        environment: str,
        request_id: str,
        actor_name: str,
        permission_level: str,
    ) -> dict | None:
        """
        Creates a GitHub Issue for the incident — or comments on an existing
        open issue if one already exists for the same service + failure category.

        Deduplication logic:
        1. Search GitHub for open issue: [CERBERUS] {service_name} + {failure_category}
        2. If found → add a comment with the new analysis (no duplicate issue)
        3. If not found → create a new issue with full incident report

        Returns dict with: number, url, title, repo, action (created|commented)
        """
        failure_category = failure.get("category", "unknown")

        # ── Step 1: Deduplication check ───────────────────────────────────────
        existing = await self._find_existing_issue(
            github_token=github_token,
            repo=repo,
            service_name=service_name,
            failure_category=failure_category,
        )

        # ── Step 2: Comment on existing issue ─────────────────────────────────
        if existing:
            comment = await self._add_comment(
                github_token=github_token,
                repo=repo,
                issue_number=existing["number"],
                remediation=remediation,
                request_id=request_id,
                permission_level=permission_level,
            )
            return {
                "number":     existing["number"],
                "url":        existing["html_url"],
                "title":      existing["title"],
                "repo":       repo,
                "action":     "commented",          # tells the frontend this was a dedup
                "comment_url": comment.get("html_url") if comment else None,
            }

        # ── Step 3: Create new issue ──────────────────────────────────────────
        return await self._create_new_issue(
            github_token=github_token,
            repo=repo,
            remediation=remediation,
            failure=failure,
            service_name=service_name,
            environment=environment,
            request_id=request_id,
            actor_name=actor_name,
            permission_level=permission_level,
        )

    async def _create_new_issue(
        self,
        github_token: str,
        repo: str,
        remediation: dict,
        failure: dict,
        service_name: str,
        environment: str,
        request_id: str,
        actor_name: str,
        permission_level: str,
    ) -> dict | None:
        """Creates a brand new GitHub Issue with the full incident report."""

        failure_category = failure.get("category", "unknown")
        severity         = remediation.get("severity", "unknown")
        confidence       = remediation.get("confidence", 0)
        command          = remediation.get("command")
        risk             = remediation.get("risk_assessment", "N/A")

        # Title includes service_name and failure_category for dedup search matching
        title = f"[CERBERUS] {service_name} — {failure_category.replace('_', ' ').title()}"

        body = f"""## 🚨 Incident Report — Auto-generated by Cerberus

| Field | Value |
|---|---|
| **Request ID** | `{request_id}` |
| **Service** | `{service_name}` |
| **Environment** | `{environment}` |
| **Severity** | `{severity.upper()}` |
| **Failure Category** | `{failure_category}` |
| **Detected By** | `{actor_name}` |
| **Permission Level** | `{permission_level}` |
| **AI Confidence** | `{confidence}%` |
| **Risk Assessment** | `{risk}/100` |
| **Est. Downtime** | `{remediation.get('estimated_downtime', 'unknown')}` |

---

## 🔍 Root Cause

{remediation.get('root_cause', 'Not determined')}

## 🧠 Agent Reasoning

{remediation.get('reasoning', 'N/A')}

## 🔒 Security Verdict

{remediation.get('security_verdict', 'N/A')}

## 💥 Blast Radius

{remediation.get('blast_radius', 'N/A')}

## 🛠 Suggested Fix

```
{remediation.get('suggested_fix', 'N/A')}
```

## ✅ Safe Alternatives (run these first)

```bash
{chr(10).join(remediation.get('safe_alternatives', ['N/A']))}
```
"""

        if command:
            body += f"""
## ⚡ Executable Command (Admin Authorized)

```bash
{command}
```
"""
        else:
            body += """
## 🔒 Executable Command

Not included — requires admin role to unlock.
"""

        body += f"""
## ⏪ Rollback Procedure

```
{remediation.get('rollback', 'N/A')}
```

---
> ♻️ **Duplicate detection active** — if this service fails again, Cerberus will comment here instead of opening a new issue.

*Auto-generated by [Cerberus](https://github.com/shubhayu-dev/Cerberus) · Vertex AI Gemini 2.5 Flash · Auth0 Token Vault · `{request_id}`*"""

        # Labels
        labels = ["incident", "cerberus", "auto-generated"]
        if sev_label := SEVERITY_LABELS.get(severity):
            labels.append(sev_label)
        if cat_label := CATEGORY_LABELS.get(failure_category):
            labels.append(cat_label)

        url = f"https://api.github.com/repos/{repo}/issues"

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.post(
                    url,
                    json={"title": title, "body": body, "labels": labels},
                    headers=GH_HEADERS(github_token),
                )
                r.raise_for_status()
                issue = r.json()
                print(
                    f"VAULT: ✅ New issue created | repo={repo} | "
                    f"issue=#{issue['number']} | url={issue['html_url']}"
                )
                return {
                    "number":  issue["number"],
                    "url":     issue["html_url"],
                    "title":   issue["title"],
                    "repo":    repo,
                    "action":  "created",
                }

        except httpx.HTTPStatusError as e:
            log.error(f"GitHub API error {e.response.status_code}: {e.response.text}")
            return None
        except Exception as e:
            log.error(f"Failed to create GitHub issue: {e}")
            return None


# Global instance
vault = TokenVault()