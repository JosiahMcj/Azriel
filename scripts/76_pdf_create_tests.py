"""Extensive pdf_create test suite.

Categories:
  A. Direct invocation (no model) -- 10 cases covering format, edges, errors
  B. Sandbox security -- 3 cases for path traversal / abs paths
  C. End-to-end via /chat -- 2 realistic prompts
  D. Round-trip with pdf_extract -- verify written content reads back

Each case logs: pass/fail + evidence (file size, byte sniff, extracted text).
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Make the repo root importable regardless of where the test is run.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Direct tool import (no model)
from azriel.tools.pdf_create import pdf_create
from azriel.tools.pdf_extract import pdf_extract

SANDBOX = Path.home() / "azriel-files"
BASE = os.environ.get("AZRIEL_URL", "http://127.0.0.1:8080")


def _basic_auth_token() -> str:
    """read Basic Auth from env first; fall back to the
    live launchd plist so the test still runs on the deployment host
    without requiring the user to re-enter credentials. Never embeds
    credentials in this file."""
    user = os.environ.get("AZRIEL_BASIC_AUTH_USER", "").strip()
    pw = os.environ.get("AZRIEL_BASIC_AUTH_PASS", "").strip()
    if not (user and pw):
        plist = Path.home() / "Library" / "LaunchAgents" / "com.azriel.server.plist"
        if plist.exists():
            try:
                for k in ("AZRIEL_BASIC_AUTH_USER", "AZRIEL_BASIC_AUTH_PASS"):
                    v = subprocess.run(
                        ["/usr/libexec/PlistBuddy", "-c",
                         f"Print :EnvironmentVariables:{k}", str(plist)],
                        capture_output=True, text=True, timeout=5,
                    ).stdout.strip()
                    if k.endswith("USER"):
                        user = v
                    else:
                        pw = v
            except Exception:
                pass
    if not (user and pw):
        return ""
    return "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()


AUTH_HEADER = _basic_auth_token()


def chat(message: str, sid: str) -> dict:
    body = json.dumps({"message": message, "session_id": sid}).encode()
    req = urllib.request.Request(
        BASE + "/chat", data=body,
        headers={"Content-Type": "application/json", "Authorization": AUTH_HEADER},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        payload = json.loads(r.read())
    payload["_elapsed"] = time.time() - t0
    return payload


def is_pdf_v14(p: Path) -> bool:
    if not p.exists():
        return False
    head = p.read_bytes()[:8]
    return head.startswith(b"%PDF-1.")


def page_count(p: Path) -> int:
    """Cheap page count: count `/Type /Page ` occurrences (not /Pages)."""
    if not p.exists():
        return 0
    raw = p.read_bytes()
    return raw.count(b"/Type /Page ") - raw.count(b"/Type /Pages")


passed = 0
failed = 0
failures: list[str] = []


def report(label: str, ok: bool, detail: str) -> None:
    global passed, failed
    marker = "PASS" if ok else "FAIL"
    print(f"[{marker}] {label} -- {detail}")
    if ok:
        passed += 1
    else:
        failed += 1
        failures.append(f"{label}: {detail}")


# ========== A. DIRECT INVOCATION ==========

print("=" * 60)
print("A. Direct invocation tests")
print("=" * 60)

# A1: simple one-page
r = pdf_create("A1_simple|Hello World\n\nThis is a basic test of the pdf_create tool.")
p = SANDBOX / "A1_simple.pdf"
report("A1_simple", is_pdf_v14(p) and "created" in r,
       f"r={r[:80]!r} valid={is_pdf_v14(p)} size={p.stat().st_size if p.exists() else 0}")

# A2: multiple paragraphs
r = pdf_create("A2_multipara|Title Line\n\nFirst paragraph of body content.\n\n"
               "Second paragraph.\n\nThird paragraph.")
p = SANDBOX / "A2_multipara.pdf"
report("A2_multipara", is_pdf_v14(p), f"size={p.stat().st_size if p.exists() else 0}")

# A3: long content -- should multi-page
long_body = "\n\n".join(
    f"Paragraph {i}: " + "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 10
    for i in range(20)
)
r = pdf_create(f"A3_multipage|Long Document Title\n\n{long_body}")
p = SANDBOX / "A3_multipage.pdf"
n_pages = page_count(p)
report("A3_multipage", is_pdf_v14(p) and n_pages >= 2,
       f"size={p.stat().st_size if p.exists() else 0} pages~={n_pages}")

# A4: parens in title (PDF string special chars)
r = pdf_create("A4_parens|Genesis (1:1)\n\nIn the beginning (literally first) God created.")
p = SANDBOX / "A4_parens.pdf"
report("A4_parens", is_pdf_v14(p),
       f"size={p.stat().st_size if p.exists() else 0}")

# A5: backslashes + quotes
r = pdf_create('A5_specials|Test \\backslash and "quotes"\n\nContent with \\n literal and (escaped) parens.')
p = SANDBOX / "A5_specials.pdf"
report("A5_specials", is_pdf_v14(p),
       f"size={p.stat().st_size if p.exists() else 0}")

# A6: unicode in title (PDF Helvetica is ASCII; unicode may render as garbled but should not crash)
r = pdf_create("A6_unicode|Café résumé naïve\n\nUnicode body with é à ü ñ.")
p = SANDBOX / "A6_unicode.pdf"
report("A6_unicode_no_crash", is_pdf_v14(p) or r.startswith("ERROR"),
       f"r={r[:80]!r} valid={is_pdf_v14(p)}")

# A7: very long single line (no word breaks) -- must wrap by char count
single = "x" * 500
r = pdf_create(f"A7_longline|Title\n\n{single}")
p = SANDBOX / "A7_longline.pdf"
report("A7_longline", is_pdf_v14(p), f"size={p.stat().st_size if p.exists() else 0}")

# A8: empty content -> ERROR
r = pdf_create("A8_empty|")
report("A8_empty_content_error", r.startswith("ERROR"), f"r={r!r}")

# A9: empty name -> ERROR
r = pdf_create("|content body here")
report("A9_empty_name_error", r.startswith("ERROR"), f"r={r!r}")

# A10: missing pipe -> ERROR
r = pdf_create("nopipe content with no separator")
report("A10_missing_pipe_error", r.startswith("ERROR"), f"r={r!r}")


# ========== B. SANDBOX SECURITY ==========

print()
print("=" * 60)
print("B. Sandbox security tests")
print("=" * 60)

# B1: path traversal via ..
r = pdf_create("../../../tmp/escape|payload")
report("B1_path_traversal_blocked", r.startswith("ERROR"), f"r={r!r}")

# B2: absolute path attempt
r = pdf_create("/tmp/escape_abs|payload")
# Behavior: lstrip('/') makes this 'tmp/escape_abs.pdf' inside the sandbox.
# That's acceptable (still sandboxed) but confirm it didn't escape.
escaped = Path("/tmp/escape_abs.pdf").exists() and Path("/tmp/escape_abs.pdf").stat().st_size > 100
report("B2_abs_path_clamped", not escaped,
       f"r={r[:80]!r} escaped_to_/tmp={escaped}")

# B3: subdirectory creation
r = pdf_create("reports/q1_summary|Q1 Summary\n\nMonthly review.")
p = SANDBOX / "reports/q1_summary.pdf"
report("B3_subdir_created", is_pdf_v14(p),
       f"size={p.stat().st_size if p.exists() else 0}")


# ========== D. ROUND-TRIP ==========

print()
print("=" * 60)
print("D. Round-trip pdf_create -> pdf_extract")
print("=" * 60)

unique = "JEHOVAH-JIREH-MARKER-" + str(int(time.time()))
r = pdf_create(f"D1_roundtrip|Round Trip Test\n\nThe secret marker is {unique}.\n\nVerifying readback.")
extracted = pdf_extract("D1_roundtrip.pdf")
report("D1_roundtrip_marker_preserved", unique in extracted,
       f"extract_len={len(extracted)} marker_in_text={unique in extracted}")

# D2: round-trip on multi-page
multi_body = "\n\n".join(
    f"Section {i}. Marker-{i}-token-XYZ. " + "Filler content here. " * 30
    for i in range(15)
)
r = pdf_create(f"D2_rt_multi|Multi Section Doc\n\n{multi_body}")
extracted2 = pdf_extract("D2_rt_multi.pdf")
markers_found = sum(1 for i in range(15) if f"Marker-{i}-token-XYZ" in extracted2)
report("D2_roundtrip_multipage", markers_found >= 12,
       f"markers_found={markers_found}/15 extract_len={len(extracted2)}")


# ========== C. END-TO-END VIA CHAT ==========

print()
print("=" * 60)
print("C. End-to-end chat tests")
print("=" * 60)

# C1: realistic sermon outline request
sid = f"pdf-test-c1-{int(time.time())}"
prompt = (
    "Use pdf_create to make a one-page PDF named 'sermon_romans828'. "
    "Title: 'Romans 8:28 -- All Things For Good'. "
    "Body should be three short paragraphs introducing the verse, the "
    "biblical context (those who love God), and a closing application "
    "for the listener. Keep each paragraph under 4 sentences."
)
r = chat(prompt, sid)
fired = "pdf_create" in [c.get("name") for c in (r.get("calls") or [])]
file_made = (SANDBOX / "sermon_romans828.pdf").exists()
report("C1_chat_sermon_pdf", fired and file_made,
       f"fired={fired} file_exists={file_made} elapsed={r['_elapsed']:.1f}s")

# C2: prayer card with structured content
sid = f"pdf-test-c2-{int(time.time())}"
prompt = (
    "Use pdf_create to make a PDF named 'prayer_card_evening'. "
    "Title: 'Evening Prayer'. Then three short paragraphs with the "
    "Lord's Prayer, a thanks paragraph, and a request paragraph."
)
r = chat(prompt, sid)
fired = "pdf_create" in [c.get("name") for c in (r.get("calls") or [])]
file_made = (SANDBOX / "prayer_card_evening.pdf").exists()
extracted_c2 = pdf_extract("prayer_card_evening.pdf") if file_made else ""
has_lords = "hallow" in extracted_c2.lower() or "father" in extracted_c2.lower()
report("C2_chat_prayer_card", fired and file_made and has_lords,
       f"fired={fired} file_exists={file_made} lords_prayer_in_text={has_lords} elapsed={r['_elapsed']:.1f}s")


# ========== SUMMARY ==========

print()
print("=" * 60)
print(f"SUMMARY: {passed} passed, {failed} failed")
print("=" * 60)
if failures:
    print("Failures:")
    for f in failures:
        print(f" - {f}")
