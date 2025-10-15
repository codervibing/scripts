import re
import argparse
from typing import List, Optional, Tuple
import pandas as pd
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer, util
import torch

# ---------- Config ----------
PARENT_COL = "Parent Group"          # exact column name for parent list
INVESTOR_COL = "Investor Name"       # exact column name for investor names
OUTPUT_COL = "Predicted_Parent"
CONF_COL = "Confidence"
METHOD_COL = "Method"
FUZZY_COL = "Fuzzy_Score"
EMB_COL = "Embed_Score"
REVIEW_COL = "Needs_Review"

# Common corporate suffixes / noise tokens to strip during normalization
SUFFIXES = [
    r"\blimited\b", r"\bltd\b", r"\bplc\b", r"\bllc\b", r"\bllp\b", r"\binc\b", r"\bco\b", r"\bcorp\b",
    r"\bsa\b", r"\bs\.a\.\b", r"\bsicav\b", r"\bse\b", r"\bag\b", r"\bspa\b", r"\bgmbh\b", r"\bsarl\b",
    r"\bpte\b", r"\bpte\.?\s+ltd\b", r"\bholdings?\b", r"\basset\s+management\b", r"\basset\s+mgt\b",
    r"\bgroup\b", r"\binternational\b", r"\bglobal\b", r"\bpartners?\b", r"\binvestments?\b", r"\bmanagement\b",
    r"\bmanagers?\b", r"\badvisors?\b", r"\bcapital\b", r"\bservices?\b", r"\btrust\b", r"\bbank\b"
]
SUFFIX_PATTERN = re.compile("|".join(SUFFIXES), flags=re.IGNORECASE)

# Optional: quick keyword rules for dead-simple high-precision hits
# Extend freely: {"Parent Name": ["aliases", "tickers", "common abbreviations", ...]}
RULES = {
    "BlackRock": ["blackrock", "ishares", "br iic"],
    "Goldman Sachs": ["gs", "gsam", "goldman"],
    "Morgan Stanley": ["ms", "msim", "morgan stanley"],
    "J.P. Morgan": ["jpm", "jpmorgan", "jp morgan", "jp morgan"],
    "UBS": ["ubs", "ubs am"],
    "Amundi": ["amundi", "cppr amundi"],
    "Vanguard": ["vanguard", "vgi"],
}

# ---------- Helpers ----------
def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r"[&/.,\-()’'`]", " ", t)           # punctuation -> space
    t = SUFFIX_PATTERN.sub(" ", t)                  # remove suffixes/noise
    t = re.sub(r"\s+", " ", t).strip()
    return t

def rule_based_match(investor_norm: str, parent_names: List[str]) -> Optional[str]:
    # Check RULES by substring contains
    for parent in parent_names:
        aliases = RULES.get(parent, [])
        for alias in aliases:
            if alias in investor_norm:
                return parent
    return None

def fuzzy_best(investor: str, parents: List[str]) -> Tuple[Optional[str], int]:
    if not parents:
        return None, 0
    match = process.extractOne(
        investor,
        parents,
        scorer=fuzz.token_set_ratio
    )
    if match:
        name, score, _ = match
        return name, int(score)
    return None, 0

def embed_best(investor: str, parent_list: List[str], model: SentenceTransformer, parent_emb: torch.Tensor) -> Tuple[Optional[str], float]:
    if not parent_list:
        return None, 0.0
    inv_emb = model.encode([investor], convert_to_tensor=True, normalize_embeddings=True)
    # cosine similarities (normalized) -> [-1,1], but here ~[0,1]
    sims = (inv_emb @ parent_emb.T)[0]
    idx = int(torch.argmax(sims).item())
    return parent_list[idx], float(sims[idx].item())

def combine_confidence(fuzzy_score: int, emb_score: float) -> int:
    """
    Combine RapidFuzz 0..100 and embedding cosine 0..1 -> 0..100.
    Simple weighted blend (tuneable).
    """
    emb_pct = emb_score * 100.0
    # Heavier weight to embeddings (often better for abbreviations), but keep fuzzy too
    combined = 0.6 * emb_pct + 0.4 * fuzzy_score
    return int(round(combined))

def decide_label(
    inv_raw: str,
    inv_norm: str,
    parents_unique: List[str],
    model: SentenceTransformer,
    parent_emb: torch.Tensor,
    high_conf: int = 85,
    med_conf: int = 70
):
    # 1) Rules first (high precision)
    rule_parent = rule_based_match(inv_norm, parents_unique)
    if rule_parent:
        # Give strong confidence if also supported by fuzzy/emb
        f_name, f_score = fuzzy_best(inv_raw, parents_unique)
        e_name, e_score = embed_best(inv_norm, parents_unique, model, parent_emb)
        conf = combine_confidence(f_score, e_score) if (f_name == rule_parent or e_name == rule_parent) else 90
        return rule_parent, conf, "rule+fusion", f_score, e_score

    # 2) Fuzzy
    f_name, f_score = fuzzy_best(inv_raw, parents_unique)

    # 3) Embeddings (use normalized strings for semantic cue)
    e_name, e_score = embed_best(inv_norm, parents_unique, model, parent_emb)

    # 4) Combine / choose best
    conf = combine_confidence(f_score, e_score)
    # Prefer the name that appears most supported. If they disagree, pick by higher individual signal.
    if f_name == e_name:
        chosen = f_name
        method = "fuzzy+embed"
    else:
        # Compare normalized signals
        if (f_score >= 92 and e_score < 0.55):   # very strong fuzzy, weak embed
            chosen, method = f_name, "fuzzy"
        elif (e_score >= 0.70 and f_score < 75): # strong embed, meh fuzzy
            chosen, method = e_name, "embed"
        else:
            # fall back to best confidence overall
            chosen = e_name if (e_score*100) >= f_score else f_name
            method = "blend"

    # Confidence thresholds can flag review later
    return chosen, conf, method, f_score, e_score

def main():
    ap = argparse.ArgumentParser(description="Map Investor Names to Parent Groups.")
    ap.add_argument("--input", required=True, help="Path to input Excel (.xlsx)")
    ap.add_argument("--output", required=True, help="Path to output Excel (.xlsx)")
    ap.add_argument("--sheet", default=0, help="Sheet name or index for input")
    ap.add_argument("--parent-col", default=PARENT_COL, help="Column name for Parent Group")
    ap.add_argument("--investor-col", default=INVESTOR_COL, help="Column name for Investor Name")
    ap.add_argument("--high-conf", type=int, default=85, help="Confidence >= this -> no review")
    ap.add_argument("--med-conf", type=int, default=70, help="Confidence < high and >= med -> review only if it disagrees with existing parent")
    args = ap.parse_args()

    df = pd.read_excel(args.input, sheet_name=args.sheet)
    if args.parent_col not in df.columns or args.investor_col not in df.columns:
        raise ValueError(f"Input must contain columns '{args.investor_col}' and '{args.parent_col}'")

    # Unique parent list (as raw + normalized sidecar)
    parents_raw = [p for p in df[args.parent_col].dropna().astype(str).unique().tolist()]
    parents_norm = [normalize(p) for p in parents_raw]

    # Load embeddings once
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    parent_emb = model.encode(parents_norm, convert_to_tensor=True, normalize_embeddings=True)

    # Prepare outputs
    preds, confs, methods, fzs, embs = [], [], [], [], []

    for inv in df[args.investor_col].astype(str).fillna(""):
        inv_norm = normalize(inv)
        parent, conf, method, fz, em = decide_label(
            inv_raw=inv,
            inv_norm=inv_norm,
            parents_unique=parents_raw,
            model=model,
            parent_emb=parent_emb,
            high_conf=args.high_conf,
            med_conf=args.med_conf
        )
        preds.append(parent)
        confs.append(conf)
        methods.append(method)
        fzs.append(fz)
        embs.append(round(em, 4))

    out = df.copy()
    out[OUTPUT_COL] = preds
    out[CONF_COL] = confs
    out[METHOD_COL] = methods
    out[FUZZY_COL] = fzs
    out[EMB_COL] = embs

    # Review logic:
    # - Confidence < med_conf  -> review
    # - med_conf <= conf < high_conf -> review if it disagrees with existing parent
    needs_review = []
    for pred, conf, existing in zip(out[OUTPUT_COL], out[CONF_COL], out[args.parent_col].astype(str).fillna("")):
        disagree = (normalize(pred) != normalize(existing)) if existing else True
        if conf < args.med_conf:
            needs_review.append(True)
        elif conf < args.high_conf and disagree:
            needs_review.append(True)
        else:
            needs_review.append(False)
    out[REVIEW_COL] = needs_review

    # Sort: review first
    out.sort_values(by=[REVIEW_COL, CONF_COL], ascending=[False, True], inplace=True)

    # Save
    out.to_excel(args.output, index=False)
    print(f"Done. Wrote {args.output}")
    print(f"Rows flagged for review: {sum(needs_review)} / {len(out)}")
    print("Tip: Inspect low-confidence rows and extend RULES for frequent subsidiaries.")
    
if __name__ == "__main__":
    main()