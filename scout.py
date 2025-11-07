def extract_sections(text: str):
    import re
    HEADERS = ["Answer", "Reasoning", "SourceRef"]

    # Trim to model output
    m = re.search(r'Assistant\s*:\s*', text, flags=re.IGNORECASE)
    if m:
        text = text[m.end():]

    headers_alt = "|".join(map(re.escape, HEADERS))
    out = {}

    for h in HEADERS:
        pat = re.compile(
            rf'(?ms)^{re.escape(h)}\s*:\s*(.*?)'
            rf'(?=^(?:{headers_alt})\s*:|^###\s*End of INST\s*###|\[/INST\]|\Z)'
        )
        matches = [m.group(1).strip() for m in pat.finditer(text)]
        if matches:
            val = matches[-1]
            if val and "<your" not in val.lower():
                out[h] = val

    # Cleanup for SourceRef
    if "SourceRef" in out:
        lines = [ln.strip() for ln in out["SourceRef"].splitlines() if ln.strip()]
        tail = [ln for ln in lines if ln.endswith(")")]
        out["SourceRef"] = tail[-1] if tail else (lines[-1] if lines else "")

    # Combine dynamically
    combined = []
    for h in HEADERS:
        if h in out:
            combined.append(f"{h}: {out[h]}")

    return "\n\n".join(combined)