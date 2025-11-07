import re

HEADERS = ["Answer", "Reasoning", "SourceRef"]

# compile once
PATTERNS = {
    h: re.compile(
        rf"(?im)^ {h} \s*:\s* (.*?) (?= ^(?:{'|'.join(HEADERS)})\s*: | \Z)",
        re.DOTALL | re.MULTILINE | re.VERBOSE,
    )
    for h in HEADERS
}

def extract_sections(text: str):
    out = {}
    for h, pat in PATTERNS.items():
        # all matches; take the final one if multiple occur
        matches = [m.group(1).strip() for m in pat.finditer(text)]
        if matches:
            out[h] = matches[-1]
    # optional: tighten SourceRef to last line ending with ')'
    if "SourceRef" in out:
        lines = [ln.rstrip() for ln in out["SourceRef"].splitlines() if ln.strip()]
        tail = [ln for ln in lines if ln.endswith(")")]
        if tail:
            out["SourceRef"] = tail[-1]
        else:
            out["SourceRef"] = lines[-1] if lines else ""
    return out