"""Test 3 of NVIDIA-NeMo/Curator#1750: reference contamination.

Apply the FineWeb-Edu quality classifier to text that should count as high quality
and report the fraction it would drop. Curator's FineWebModelStage labels a document
low_quality when score < 2.5, and the model card's recommended curation threshold
(int_score >= 3) is the same cut, so one threshold covers both.

Sets:
  wikipedia       English Wikipedia (20231101), not part of the classifier's training data
  annotated_hi    web samples Llama3 scored 4-5; may overlap the classifier's training set
  annotated_lo    web samples Llama3 scored 0-1; control, the classifier should drop most
  openstax        Cosmopedia synthetic textbook sections in the OpenStax style; educational by
                  construction and outside FineWeb, so outside the classifier's training data
  khanacademy     Cosmopedia synthetic Khan Academy style lessons; same reasoning

Caveat on wikipedia: the classifier scores grade-school educational value, and a random
article is often a biography or a place stub, so a high drop rate there is largely by design.
The informative sets are annotated_hi (its own target) and the two textbook-style sets.
"""
import json, math, random, sys, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SEED = 0
THRESH = 2.5
MODEL = "HuggingFaceFW/fineweb-edu-classifier"
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
random.seed(SEED)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def wikipedia(n):
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    ds = ds.shuffle(seed=SEED, buffer_size=20000)
    out = []
    for row in ds:
        if len(row["text"]) >= 500:  # skip stubs and redirects
            out.append(row["text"])
        if len(out) >= n:
            break
    return out


def annotated(n, lo, hi):
    ds = load_dataset("HuggingFaceFW/fineweb-edu-llama3-annotations", split="train", streaming=True)
    ds = ds.shuffle(seed=SEED, buffer_size=50000)
    out = []
    for row in ds:
        if lo <= row["score"] <= hi:
            out.append(row["text"])
        if len(out) >= n:
            break
    return out


def cosmopedia(n, subset):
    ds = load_dataset("HuggingFaceTB/cosmopedia", subset, split="train", streaming=True)
    ds = ds.shuffle(seed=SEED, buffer_size=10000)
    out = []
    for row in ds:
        out.append(row["text"])
        if len(out) >= n:
            break
    return out


def score(texts, tok, model, bs=16):
    s = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding="longest", truncation=True, max_length=512).to(DEV)
        with torch.no_grad():
            s.extend(model(**enc).logits.squeeze(-1).float().cpu().tolist())
    return np.array(s)


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(DEV).eval()
    sets = {
        "wikipedia": wikipedia(N),
        "annotated_hi": annotated(N, 4, 5),
        "annotated_lo": annotated(N, 0, 1),
        "openstax": cosmopedia(N, "openstax"),
        "khanacademy": cosmopedia(N, "khanacademy"),
    }
    res = {"model": MODEL, "threshold": THRESH, "n_per_set": N, "seed": SEED, "device": str(DEV), "sets": {}}
    for name, texts in sets.items():
        sc = score(texts, tok, model)
        k = int((sc < THRESH).sum())
        lo, hi = wilson(k, len(sc))
        res["sets"][name] = {
            "n": len(sc), "dropped": k, "drop_rate": k / len(sc), "wilson95": [lo, hi],
            "score_mean": float(sc.mean()), "score_median": float(np.median(sc)),
            "score_quantiles": {q: float(np.quantile(sc, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        }
        print(f"{name:13s} n={len(sc):5d} drop={k/len(sc):.3f} [{lo:.3f},{hi:.3f}] median={np.median(sc):.2f}", flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open("results_test3.json", "w"), indent=2)
    print("wrote results_test3.json", res["seconds"], "s")


if __name__ == "__main__":
    main()
