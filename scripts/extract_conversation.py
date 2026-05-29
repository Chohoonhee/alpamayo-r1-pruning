"""Extract human-readable conversation transcript from Claude Code JSONL.

Reads the session's .jsonl log file from ~/.claude/projects/, filters to
user prompts and assistant text replies (drops tool calls / results / system
reminders), and writes a markdown summary to docs/CONVERSATION.md.

The full JSONL is too large and noisy to commit; this gives the user the
readable narrative so they can see what was discussed and why decisions
were made, alongside the experiment artifacts.

Usage:
    python extract_conversation.py [--src PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

DEFAULT_SRC = Path(
    "/home/irteam/.claude/projects/-home-irteam-ws/"
    "28a62f35-3336-49d9-9f78-49264856273c.jsonl"
)
DEFAULT_OUT = Path("/home/irteam/ws/alpamayo_pruning_share/docs/CONVERSATION.md")


def iter_messages(path: Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


_SECRET_PATTERNS = [
    # GitHub PAT (classic ghp_, fine-grained github_pat_, and legacy gh*)
    (r"ghp_[A-Za-z0-9]{20,}", "[REDACTED-GITHUB-PAT]"),
    (r"github_pat_[A-Za-z0-9_]{20,}", "[REDACTED-GITHUB-PAT]"),
    (r"gho_[A-Za-z0-9]{20,}", "[REDACTED-GITHUB-OAUTH]"),
    (r"ghu_[A-Za-z0-9]{20,}", "[REDACTED-GITHUB-USER-OAUTH]"),
    (r"ghs_[A-Za-z0-9]{20,}", "[REDACTED-GITHUB-SERVER]"),
    # Hugging Face user access token (hf_...)
    (r"hf_[A-Za-z0-9]{20,}", "[REDACTED-HF-TOKEN]"),
    # Anthropic API key (sk-ant-...)
    (r"sk-ant-[A-Za-z0-9_\-]{20,}", "[REDACTED-ANTHROPIC-KEY]"),
    # Generic OpenAI sk-... (less specific, last)
    (r"\bsk-[A-Za-z0-9]{40,}", "[REDACTED-API-KEY]"),
    # AWS access key
    (r"AKIA[A-Z0-9]{16}", "[REDACTED-AWS-AK]"),
    # Long bearer-style hex/base64 (32+ chars) — heuristic, may over-match
    # (kept narrow with delimiters to avoid clobbering experiment outputs)
]


def _scrub_secrets(txt: str) -> str:
    """Redact known secret patterns from a string."""
    import re
    for pat, repl in _SECRET_PATTERNS:
        txt = re.sub(pat, repl, txt)
    return txt


def _strip_reminders(txt: str) -> str:
    """Drop <system-reminder>…</system-reminder> and <task-notification>…
    blocks from a text block, since they're harness noise, not content.
    Also scrubs known secret patterns."""
    import re
    txt = re.sub(r"<system-reminder>.*?</system-reminder>", "", txt, flags=re.DOTALL)
    txt = re.sub(r"<task-notification>.*?</task-notification>", "", txt, flags=re.DOTALL)
    txt = re.sub(r"<command-name>.*?</command-name>", "", txt, flags=re.DOTALL)
    txt = re.sub(r"<command-message>.*?</command-message>", "", txt, flags=re.DOTALL)
    txt = re.sub(r"<local-command-stdout>.*?</local-command-stdout>", "", txt, flags=re.DOTALL)
    return _scrub_secrets(txt.strip())


def extract_text(content) -> str:
    """Pull out user-facing text from a message content list/str."""
    if isinstance(content, str):
        return _strip_reminders(content)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                bt = block.get("type")
                if bt == "text":
                    parts.append(_strip_reminders(block.get("text", "")))
                elif bt == "tool_use":
                    name = block.get("name", "?")
                    parts.append(f"_[tool call: {name}]_")
                elif bt == "tool_result":
                    pass
        return "\n".join(p for p in parts if p.strip())
    return ""


def is_real_user_text(text: str) -> bool:
    """Drop system reminders, tool-result wrappers, and task notifications."""
    if not text or not text.strip():
        return False
    t = text.strip()
    if t.startswith("<system-reminder>") or t.startswith("<task-notification>"):
        return False
    if t.startswith("Caveat:") or t.startswith("[Request"):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--max_bytes", type=int, default=2_000_000,
                    help="cap output size (default 2 MB).")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        print(f"[err] {src} not found")
        return

    md = [
        "# Conversation — alignment-grounded pruning session",
        "",
        "Extracted by `scripts/extract_conversation.py` from the Claude Code",
        f"session log (`{src.name}`). Tool calls and results are stripped;",
        "only the user / assistant message text is preserved. Long passages",
        "may be truncated for size — the original session log remains the",
        "full record.",
        "",
        "---",
        "",
    ]
    size = sum(len(s) for s in md)

    last_role = None
    for msg in iter_messages(src):
        if msg.get("type") not in ("user", "assistant", None):
            continue
        message_obj = msg.get("message", msg)
        role = message_obj.get("role")
        if role not in ("user", "assistant"):
            continue
        content = message_obj.get("content", "")
        text = extract_text(content).strip()
        if role == "user" and not is_real_user_text(text):
            continue
        if not text:
            continue

        header = "## User" if role == "user" else "### Assistant"
        chunk = f"{header}\n\n{text}\n\n"
        if size + len(chunk) > args.max_bytes:
            md.append("\n*[Output truncated to fit size budget.]*\n")
            break
        md.append(chunk)
        size += len(chunk)
        last_role = role

    with open(out, "w") as f:
        f.write("".join(md))
    print(f"Wrote {out} ({size} bytes)")


if __name__ == "__main__":
    main()
