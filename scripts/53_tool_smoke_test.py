"""Tool smoke test -- exercise every tool in the registry once.

For each entry in REGISTRY, call the tool with a representative argument
and report PASS / FAIL plus a 200-char preview. A tool counts as PASS if
the call returns a non-error string. The github_query and cloudflare_query
entries are expected to ERROR until auth is configured -- they're listed
under EXPECTED_AUTH_FAIL so a missing-auth message counts as a known
limitation rather than a regression.

Run on a development machine:
    cd ~/azriel-arch && PYTHONPATH=. ~/.azriel/.venv/bin/python \
        scripts/53_tool_smoke_test.py
"""
import time
from azriel.tools import REGISTRY, call

# (tool_name, arg)
PROBES = {
    "bible_lookup": "John 3:16",
    "crossref_lookup": "John 3:16",
    "memory_search": "Pentecost",
    "strongs_lookup": "H1",
    "web_search": "Pentecost meaning",
    "web_fetch": "https://en.wikipedia.org/wiki/Pentecost",
    "conversation_search": "test",
    "memory_insert": "smoke-test note (from 53_tool_smoke_test.py)",
    "image_search": "Jerusalem old city",
    "pdf_extract": "missler/65_Jude/65_Jude_Commentary_Handbook.pdf|1",
    "fs_list": ".",
    "fs_read": "hello.txt",
    "fs_write": "smoke/probe.txt|smoke probe",
    "weather": "Phoenix",
    "document_create": "docx|smoke-from-test|Hello from the smoke test.",
    "visualize": '<svg viewBox="0 0 32 32"><circle cx="16" cy="16" r="12" fill="#d97757"/></svg>',
    "github_query": "user:karpathy nanoGPT",
    "cloudflare_query": "zones",
}

EXPECTED_AUTH_FAIL = {"github_query", "cloudflare_query"}

OK = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
KNOWN = "\033[33mEXPECTED-FAIL (auth)\033[0m"


def run():
    print(f"running smoke test against {len(REGISTRY)} tools\n")
    fails = []
    for name in REGISTRY:
        arg = PROBES.get(name, "")
        if name not in PROBES:
            print(f" {name}: SKIP (no probe defined)")
            continue
        t0 = time.time()
        try:
            result = call(name, arg)
        except Exception as e:
            result = f"ERROR: {type(e).__name__}: {e}"
        dt = time.time() - t0
        is_err = isinstance(result, str) and result.startswith("ERROR")
        if is_err and name in EXPECTED_AUTH_FAIL:
            label = KNOWN
        elif is_err:
            label = FAIL
            fails.append((name, result))
        else:
            label = OK
        preview = (result or "")[:200].replace("\n", " ")
        print(f" {label} {name:22s} ({dt:5.2f}s) {preview}")
    print()
    if fails:
        print(f"{len(fails)} unexpected failure(s):")
        for n, r in fails:
            print(f" - {n}: {r[:300]}")
    else:
        print("all tools healthy (auth-gated tools excluded).")


if __name__ == "__main__":
    run()
