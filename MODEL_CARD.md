---
license: apache-2.0
base_model: convaiinnovations/laya
language:
  - en
datasets:
  - clinc_oos
library_name: laya-escalate
tags:
  - calibration
  - uncertainty
  - selective-prediction
  - conformal-prediction
  - decision-models
  - system-one
  - evaluation
---

# Laya `choice:11+` calibration audit — what one coarse bucket costs, and the floats that fix it

> **What this is.** A measurement of [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya)
> over 12,000 tagged intent decisions, a reproducible diagnostic, and a seven-float
> recalibration you can apply without retraining.
>
> **What this is not.** Not a finetune, not a new checkpoint, and not a claim of a better
> model. No weights are modified. It is an evaluation artifact with a fix attached.

## The finding

Laya chooses a temperature from a `(question type, option-count)` bucket. The buckets are
`{2, 3-5, 6-10, 11+}`, so **a 12-option question and a 64-option question share one
temperature**. `choice:11+` ships `0.1006`, which Laya's own loader rejects:

```
RuntimeWarning: laya: this checkpoint ships temperatures outside [0.5, 5]
which would distort confidence; clamping choice:11+=0.1006.
Treat confidence from the affected buckets as uncalibrated.
```

It is replaced with `0.5`. Across that one bucket, measured on the same 2,000 items at
every option count:

| k | accuracy | mean published `p_top1` | bias | ECE | published as exactly 1.0000 |
|---:|---:|---:|---:|---:|---:|
| 12 | 0.978 | 0.994 | +0.0160 | 0.0166 | **92.2%** |
| 20 | 0.967 | 0.990 | +0.0227 | 0.0232 | 90.3% |
| 32 | 0.954 | 0.987 | +0.0333 | 0.0346 | 85.2% |
| 48 | 0.904 | 0.970 | +0.0658 | 0.0666 | 73.5% |
| 64 | **0.797** | 0.921 | **+0.1242** | **0.1274** | 54.4% |

Accuracy falls **0.978 → 0.797** and overconfidence grows **+0.016 → +0.124** — a 7.8×
swing in bias — while one temperature serves the whole range. Paired tests (same items at
every `k`): accuracy `spearman = −1.000`, bias `spearman = +1.000`, permutation
`p = 0.0002` for both. A smooth ramp, not a cliff.

**Laya publishes certainty on 92% of 12-option questions and 54% of 64-option questions,
being right 98% and 80% of the time respectively.**

### The field this model card documents is worse than `p_top1`

That table audits `p_top1`, which you only get by reading the logits. The field Laya
documents is `confidence` — normalised entropy, `1 − H(p) / log k`, **not** `p_top1` — and
it is a different curve with a worse defect. Audited separately on 2,500 held-out
`choice:11+` decisions, 500 per option count:

| k | accuracy | mean `confidence` | bias | ECE |
|---:|---:|---:|---:|---:|
| 12 | 0.984 | 0.9932 | +0.0092 | 0.0126 |
| 20 | 0.974 | 0.9910 | +0.0170 | 0.0186 |
| 32 | 0.960 | 0.9920 | +0.0320 | 0.0355 |
| 48 | 0.920 | 0.9803 | +0.0603 | 0.0659 |
| 64 | **0.784** | **0.9395** | **+0.1555** | **0.1555** |

A **16.9x** monotone bias ramp inside one temperature cell — and bias tracks ECE closely at
every option count (equal at `k = 64`), so the error is predominantly a level shift rather
than a shape error. One fitted global temperature (`T = 1.592`; two independent objectives
agree — `1.583` from top-1 log-loss, `1.592` from matching mean confidence to accuracy)
cuts the worst bias
`0.1555 → 0.0393`, with **no decision changed**, since `softmax(z / T)` is monotone. The
shipped `0.1006` is out of the loader's accepted range and replaced by `0.5` — roughly a
third of what the field wants.

It repairs the level, not the risk. At each field's own 5% threshold the per-`k` risk
spread inside `choice:11+` moves `18.2x → 13.3x`, because accuracy itself runs `0.784 →
0.984` over those counts; no confidence scale can equalise risk across them. Fitted per
option count *to this field*, the temperatures spread **2.07x** — the same coarse-bucket
defect, visible in the documented field rather than in `p_top1`.

`artifacts/confidence_audit.json` · `scripts/audit_confidence.py`

## The fix

The repair is **not** a finer temperature grid. It is an affine recalibration of the
published top-1 probability — one slope per option count plus a single shared offset,
seven numbers:

```
p_cal = sigmoid( slope[k] * logit(p_top1) + offset )
```

Fitted on a training split and measured on held-out rows:

| scorer | params | AURC | cov @1% | @2% | @5% | global ECE | global Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`p_top1` + per-k slope + one shared offset** | **7** | **0.0097** | **0.737** | 0.888 | 0.976 | 0.0126 | **0.0407** |
| one temperature per option count (no offset) | 6 | 0.0100 | 0.773 | 0.885 | 0.976 | — | — |
| escalation head | 5,543 | 0.0106 | 0.701 | 0.891 | 0.975 | **0.0123** | 0.0413 |
| `p_top1` (as published) | 0 | 0.0128 | 0.000 | 0.748 | 0.975 | 0.0398 | 0.0487 |
| one temperature for the whole bucket | 1 | 0.0135 | 0.589 | 0.727 | 0.976 | — | — |
| Laya `confidence` | 0 | 0.0138 | 0.640 | 0.729 | 0.967 | 0.0361 | 0.0531 |
| Laya `act_probability` | 0 | 0.0222 | 0.000 | 0.000 | 0.000 | — | — |

**Not one decision changes.** The map is monotone in `p_top1` and acts on the top-1 score
alone, so Laya's chosen answer and every accuracy number are untouched. `>20`-option bias
goes `+0.0754 → +0.0168` and its ECE `0.0768 → 0.0260`, from seven numbers, with no
retraining and no new weights.

The fitted slopes on `logit(p_top1)` spread **7.82×** across option counts (`k=4: 1.974`,
`12: 0.434`, `20: 0.394`, `32: 0.378`, `48: 0.307`, `64: 0.252`) and **1.72×** within
`choice:11+` alone.

### The obvious guess — a finer temperature grid — is not the fix

The bucket is too coarse, so the natural repair is Laya's own mechanism with more buckets:
one temperature per option count. Fitted that way they come out at `1.069 / 1.415 / 1.558 /
1.554 / 1.579 / 1.636` — a **1.53×** spread overall and only **1.16×** inside `choice:11+`.
Measured inside the bucket (n = 2,500 held out):

| form | params | ECE | Brier | bias |
|---|---:|---:|---:|---:|
| `p_top1` as published | 0 | 0.0515 | 0.0561 | +0.0504 |
| one temperature, whole bucket | 1 | 0.0326 | 0.0531 | −0.0048 |
| one temperature per option count | 6 | 0.0338 | 0.0530 | −0.0049 |
| **per-k slope + one shared offset** | **7** | **0.0147** | 0.0467 | +0.0101 |
| escalation head | 5,543 | 0.0155 | 0.0476 | +0.0117 |

Six temperatures buy **nothing** over one. The reason is structural: a temperature is a
slope *through the origin* on the logit scale, and the data wants an offset of **−0.234**.
Forcing the origin compresses every fitted slope toward 1 — which is why the temperature
spread (`1.53×`) looks so much smaller than the slope spread (`7.82×`). Add one shared
offset and the same six slopes cut ECE a further **3.5×**. So **no re-tuning of Laya's
temperature grid can repair this**: the mechanism is the constraint. The temperature grid
ships in `temperatures.json` under `negative_control`, because it is the guess everyone
will make.

## The learned head was the original premise, and it is a negative result

The companion 5,543-parameter MLP over 18 distribution-shape features **loses** to the
seven floats on AURC (`0.0106` vs `0.0097`), coverage @1% (`0.701` vs `0.737`) and Brier
(`0.0413` vs `0.0407`), and only ties on coverage @2% and ECE. The correct functional form
beat the larger model class. It is shipped as a documented negative, not as the fix.

Its apparent coverage edge is an artifact of the baseline it is compared against. `p_top1`
as published scores `0.000` at 1% target risk because 79% of `choice:11+` decisions sit on
exactly `1.0000` — a tie block no threshold can split. Fit *either* per-option-count
recalibration and that block resolves into a usable ordering above the head's `0.701`
(`0.737` affine, `0.773` per-k temperature). The head's edge is real only *within* one
option count, where a temperature is a single map and therefore cannot reorder anything:
at fixed `k = 32` it reaches `0.316` coverage where the best temperature gets `0.000`.

## What remains unfixed, stated plainly

Holding `k` fixed removes every handle a temperature has. Two sweeps at `k = 32` only —
same items, same budget, same applied temperature, differing *only* in distractor
confusability — move accuracy `0.943 → 0.896` and ECE `0.0443 → 0.0806`. After the best
single recalibration, residual bias is monotone in confusability and **changes sign**
(`−0.0178 → +0.0308`, `p = 0.005`, ideal slopes spread `1.41×`, `p = 0.005`).

So the per-`k` correction fixes the dominant, `k`-driven part — not all of it. The
remainder needs a signal a scalar rescaling cannot see.

## Mechanism: cardinality, not token starvation

Not truncation. Replaying the exact option draws through Laya's real `build_sequence`:
per-option cost is **flat at ~7.4 tokens** at every `k`, nothing is truncated for
`k <= 48` (which is where accuracy has already fallen to 0.904), and only `k = 64`
overflows — dropping 27.7% of option tokens while the instructions and the customer
message survive intact. The collapse is difficulty discriminating among many similar short
labels.

## Two cautions the artifacts demonstrate

* **Sample size changes the conclusion.** On a 200-item sweep the per-`k` slope spread is
  `1.86×` against a null of `[1.09, 2.29] → p = 0.104`: "within noise". On 2,000 items the
  same statistic is `1.58×` against `[1.04, 1.17] → p = 0.005`. The effect got *smaller*;
  the null got tighter. The earlier reading was published and then overturned.
* **A conformal bound is only as good as its calibration set.** At 250 calibration rows the
  `α = 0.010` bound is violated 2.4× on held-out data (`0.0240` measured); at 3,000 rows it
  holds (`0.0063`). The code reports the deployed threshold on rows it was not fitted on
  and warns explicitly when risk exceeds the bound.

## Using the fix

```python
import json, math

fit = json.load(open("artifacts/real_big/temperatures.json"))
# "form":              "affine on the logit scale ..."
# "slope_by_options":  {"4": 1.974, "12": 0.434, "20": 0.394,
#                       "32": 0.378, "48": 0.307, "64": 0.252}
# "shared_offset":     -0.234
# "n_parameters":      7

slope = fit["slope_by_options"][str(n_options)]
offset = fit["shared_offset"]

z = math.log(p_top1 / (1.0 - p_top1))   # logit of the published top-1
p = 1.0 / (1.0 + math.exp(-(slope * z + offset)))
```

Guard the two endpoints: `p_top1 == 1.0` sends `z` to `+inf` and `p_top1 == 0.0` sends it
to `-inf`, which is fine for the logistic but not for the `log` — clamp first, or use
`math.log(p / (1 - p))` on `min(max(p, 1e-12), 1 - 1e-12)`.

The same file keeps the per-option-count **temperature** grid under `negative_control`.
Do not reach for it: it is the form that does *not* work, kept so the negative is
reproducible.

The floats are fit to `clinc_oos/plus` English intents and this checkpoint. **Refit them
before trusting them on your own traffic** — `scripts/train_escalate.py` in the repository
produces the fit from a collected sweep and writes this file.

## Limitations

1. **One corpus.** `clinc_oos/plus`, 150 short snake_case intent names. Option text is
   short, which is why the token budget never binds below `k = 64`. A corpus with long
   option descriptions would exercise `head_max_len` differently.
2. **One checkpoint and one runtime.** Laya English, bf16, CUDA, batch size 1. The clamp is
   a property of the shipped config and should transfer to the multilingual and
   typed-decisions checkpoints; the fitted floats are corpus-specific.
3. **The confusability ranking is over intent names**, a cheap proxy for semantic hard
   negatives. It moves the measured mean by 4.8× and the effect is significant, but the
   per-row achieved similarity is recorded so a stronger selector can be substituted.
4. **`act_probability` being constant at `1.0000` is a property of this distribution**, not
   a general claim that Laya's act head is broken.
5. **A single global confidence gate distributes risk unevenly**: at 5% pooled selective
   risk the per-`k` risk ranges from 0.36% to 15.0% — a 42× spread (136.6× on
   `confidence`). That figure spans the whole sweep including `k = 4`; the `confidence`
   audit above is `choice:11+` only, which is why its spread reads 18.2×. They measure
   different populations.
6. **The head's coverage @1% edge in the saturated fixed-`k` regime** (0.316 vs 0.000) is
   measured on 1,000 held-out rows where 79% of published probabilities are exactly
   `1.0000` and therefore tied. Treat it as an indication of a tie-breaking effect, not a
   validated gate.

## Files

```
artifacts/real_big/temperatures.json   the seven fitted floats (six slopes + one shared
                                       offset), plus the per-option-count temperature grid
                                       under `negative_control`
artifacts/sweep_big.npz                12,000 real decisions (the primary evidence)
artifacts/sweep_big_t1.npz             the T = 1.0 ablation, same 12,000 items, 0 changed
artifacts/sweep_big_analysis.json      full diagnostic incl. null-calibrated significance
artifacts/option_tokens.json           measured option-token counts per k (the mechanism)
artifacts/confidence_audit.json        held-out audit of the documented `confidence` field
artifacts/sweep_k32_*.npz              the fixed-k = 32 confusability experiment
artifacts/sweep_small*.npz             the 1,000-decision sweep, kept because it was wrong
artifacts/analysis_big.txt             human-readable diagnostic output
artifacts/real_big/                    the trained head, and the report that retires it
```

Apache-2.0, matching the model this work measures.
