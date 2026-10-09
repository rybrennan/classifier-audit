# classifier-audit

Does a data-quality classifier keep the documents it is supposed to keep?

This repository measures that for the public [FineWeb-Edu classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier). It implements the first and cheapest of the three tests proposed in [NVIDIA-NeMo/Curator#1750](https://github.com/NVIDIA-NeMo/Curator/issues/1750): **reference contamination**, the fraction of known-good text the classifier would drop.

The motivation is [Nait Saada et al., "Removing Noise, not Finding Gold"](https://arxiv.org/abs/2510.00866) (arXiv 2510.00866), which finds that classifier-based quality filtering also filters the high-quality set it was meant to find.

## Result

FineWeb-Edu classifier, cutoff score < 2.5 (NeMo Curator's `low_quality` label, and the same cut as the model card's recommended `int_score >= 3`). 1,000 documents per set, seed 0, first 512 tokens of each document. 95% Wilson intervals.

| Set | Dropped | 95% interval | Median score |
|---|---|---|---|
| Samples its own annotator (Llama 3 70B) rated 4-5 | **10.7%** | 8.9% to 12.8% | 3.28 |
| College-level textbook sections (Cosmopedia, OpenStax style) | **21.7%** | 19.3% to 24.4% | 3.29 |
| Grade-school lessons (Cosmopedia, Khan Academy style) | 1.9% | 1.2% to 2.9% | 4.28 |
| Random English Wikipedia, 500+ characters | 82.3% | 79.8% to 84.5% | 1.59 |
| Samples its own annotator rated 0-1 (control) | 99.9% | 99.4% to 100% | 0.99 |

**Reading it.**

- *The 10.7% reproduces the published error rate; it is not a new finding.* The model card's own hold-out confusion matrix shows 120 of 932 samples rated 4-5 predicted below 3 (12.9%). Matching that number is evidence the harness measures what it claims to.
- *The textbook gap is the new measurement.* The classifier drops about one in five college-level textbook sections against one in fifty grade-school lessons. The model card says the classifier is "focused on educational content relevant to primary and grade school levels and may not perform as well on content intended for higher education," but gives no number. This puts one on it.
- *Preliminary, not a verdict.* Without independent human judgments of the textbook sections, this characterizes how the classifier behaves across domains; it does not by itself show the classifier is miscalibrated.

**Not a finding.** Wikipedia's 82% is mostly by design: the classifier scores educational value, and a random article is often a biography or a place stub. It is reported for completeness, not as contamination.

## Caveats

- One classifier, one threshold.
- The annotated sets come from [`fineweb-edu-llama3-annotations`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-llama3-annotations), which is published as a single training split, so some samples may be in the classifier's training data. If so, the 10.7% understates the drop on unseen text.
- The textbook sets are synthetic ([Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia), generated text), chosen because they are educational by construction and outside FineWeb.
- FineWeb-Edu is a regression on LLM labels, not the binary classifier-versus-reference setup in the paper, so this is the Test 3 analog rather than the paper's exact protocol.
- Only the first 512 tokens of each document are scored, as in the model card's usage.

## Run it

```
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python test3_reference_contamination.py 1000
```

Writes `results_test3.json`. About 4 minutes on an Apple M-series laptop (MPS); CPU works, slower. Streams all data from the Hugging Face Hub; nothing is downloaded in full.

## Next

Tests 2 (Spearman correlation against human ratings) and 1 (perplexity of a proxy model trained on filtered versus unfiltered data, on the reference corpus) from the same issue, implemented inside NeMo Curator if the maintainers want it there.

## License

MIT. Copyright (c) 2026 Anchor 163 LLC.
