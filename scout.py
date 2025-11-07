import re

HEADERS = ["Answer", "Reasoning", "SourceRef"]

def extract_sections(text: str):
    # narrow to region after 'Assistant:'
    m = re.search(r'Assistant\s*:\s*', text)
    if m:
        text = text[m.end():]

    patterns = {
        h: re.compile(
            rf"(?im)^ {h} \s*:\s* (.*?) (?= ^(?:{'|'.join(HEADERS)})\s*: | ^###\s*End of INST\s*### | \[/INST\] | \Z)",
            re.DOTALL | re.MULTILINE | re.VERBOSE,
        )
        for h in HEADERS
    }

    out = {}
    for h, pat in patterns.items():
        matches = [m.group(1).strip() for m in pat.finditer(text)]
        if matches:
            val = matches[-1]
            if val and "<your" not in val.lower():
                out[h] = val

    # optional post-cleanup for SourceRef
    if "SourceRef" in out:
        lines = [ln.strip() for ln in out["SourceRef"].splitlines() if ln.strip()]
        tail = [ln for ln in lines if ln.endswith(")")]
        out["SourceRef"] = tail[-1] if tail else (lines[-1] if lines else "")

    return out