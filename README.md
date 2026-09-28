# Laya's `choice:11+` bucket holds every question from 11 to 64+ options, and gives them one temperature

A measurement of [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya)
across **12,000 tagged intent decisions**, plus a reproducible diagnostic and the few
floats that repair the worst of what it finds.

Laya scores a choice question by dividing its logits by a temperature chosen from a
`(question type, option-count)` bucket. The buckets are `{2, 3-5, 6-10, 11+}` — so a
12-option question and a 64-option question share **one** temperature. This repository
measures what that costs in both fields Laya exposes — the documented `confidence`
(a **16.9x** overconfidence ramp) and the underlying `p_top1` (a **7.8x** one) — and shows
what one temperature per option count pays back, and what it does not.

Everything below is measured on real Laya runs. Every number is re-derivable on CPU
from the committed files in [`artifacts/`](artifacts/) — see
[Reproduce](#reproduce).

---

## 1. The bucket's own loader says it is broken

Loading this checkpoint prints:

```
RuntimeWarning: laya: this checkpoint ships temperatures outside [0.5, 5]
which would distort confidence; clamping choice:11+=0.1006.
Treat confidence from the affected buckets as uncalibrated.
```

`choice:11+` ships `0.1006` — a temperature that multiplies the logits by ten — and
Laya's loader refuses to apply it, substituting `0.5`. Every question with 11 or more
options is therefore served by a single clamped constant.

## 2. What that constant publishes

Laya rounds probabilities to four decimals. At `T = 0.5` a large share of decisions
come out as **exactly `1.0000`**:

| k | bucket | accuracy | mean published `p_top1` | bias | ECE | published as exactly 1.0000 |
|---:|---|---:|---:|---:|---:|---:|
| 4 | `choice:3-5` | 0.990 | 0.969 | −0.0200 | 0.0201 | 9.8% |
| **12** | **`choice:11+`** | **0.978** | **0.994** | **+0.0160** | **0.0166** | **92.2%** |
| 20 | `choice:11+` | 0.967 | 0.990 | +0.0227 | 0.0232 | 90.3% |
| 32 | `choice:11+` | 0.954 | 0.987 | +0.0333 | 0.0346 | 85.2% |
| 48 | `choice:11+` | 0.904 | 0.970 | +0.0658 | 0.0666 | 73.5% |
| **64** | **`choice:11+`** | **0.797** | **0.921** | **+0.1242** | **0.1274** | **54.4%** |

Read the first and last rows of that block together. Inside one bucket — one
temperature, one clamp decision — accuracy falls **0.978 → 0.797** and
overconfidence grows **+0.016 → +0.124**, a **7.8x** swing in bias. Within `choice:11+`
the trends are monotone, and paired (the same items were asked at every `k`):
accuracy `spearman = −1.000`, bias `spearman = +1.000`, paired permutation
`p = 0.0002` for both. k=64 accuracy CI `[0.779, 0.814]`. It is a smooth ramp, not a
cliff, which is exactly why a finer temperature grid interpolates it.

The user-visible version: **Laya reports certainty on 92% of 12-option questions and on
54% of 64-option questions, where it is right 98% and 80% of the time respectively.**

## 3. The clamp is a second, independent defect

Re-running the identical 12,000 items with every temperature forced to `1.0`:

* **0 of 12,000 decisions changed.** `softmax(z / T)` is a positive rescaling, so
  `argmax` — and therefore every accuracy number — is invariant. That is a check on
  the pairing, and it holds exactly.
* Global ECE `0.0405 → 0.0299`, bias `+0.0403 → +0.0160`, NLL `0.3820 → 0.2144`.
* The bias *trend* survives untouched: `spearman = +1.000`, `p = 0.0002`. So the drift
  with cardinality is Laya's own confidence curve, **not** the clamp.
* But the clamp still harms. Published-certainty share at `T = 0.5` vs `T = 1.0`:
  `k=12: 92.2% → 74.3%`, `k=32: 85.2% → 59.7%`, `k=64: 54.4% → 24.6%`. Setting
  `T = 1.0` is free and strictly better, and it is still not enough — 74% of
  12-option questions remain published as certain.

## 4. Seven floats fix it — but they have to be the right seven

The repair is **not** a finer temperature grid. It is an affine recalibration of the
published top-1 probability — one slope per option count plus a single shared offset:

```
p_cal = sigmoid( slope[k] * logit(p_top1) + offset )
```

Seven numbers. Fitted on the training split and measured on held-out rows — exactly like
the head, so neither gets an advantage:

| scorer | params | AURC | coverage @1% | @2% | @5% | global ECE | global Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`p_top1` + per-k slope + one shared offset** | **7** | **0.0097** | **0.737** | 0.888 | 0.976 | 0.0126 | **0.0407** |
| one temperature per option count (no offset) | 6 | 0.0100 | 0.773 | 0.885 | 0.976 | — | — |
| escalation head | 5,543 | 0.0106 | 0.701 | 0.891 | 0.975 | **0.0123** | 0.0413 |
| `p_top1` (what Laya publishes) | 0 | 0.0128 | 0.000 | 0.748 | 0.975 | 0.0398 | 0.0487 |
| one temperature for the whole bucket | 1 | 0.0135 | 0.589 | 0.727 | 0.976 | — | — |
| Laya `confidence` | 0 | 0.0138 | 0.640 | 0.729 | 0.967 | 0.0361 | 0.0531 |
| Laya `act_probability` | 0 | 0.0222 | 0.000 | 0.000 | 0.000 | — | — |

3,000 held-out decisions, base accuracy 0.9353. Per bucket:

| bucket | n | accuracy | ECE before | ECE after | bias before | bias after |
|---|---:|---:|---:|---:|---:|---:|
| `<=4` | 500 | 0.990 | 0.0206 | 0.0118 | −0.0206 | −0.0028 |
| `11-20` | 1000 | 0.979 | 0.0151 | 0.0104 | +0.0130 | −0.0000 |
| `>20` | 1500 | 0.888 | 0.0768 | **0.0260** | +0.0754 | **+0.0168** |

Zero decisions change, no retraining, no new weights, no deployment change beyond a
`dict` of seven numbers — a slope per option count and one shared offset. The fitted
slopes on `logit(p_top1)` spread **7.82x** across all six option counts (`k=4: 1.974`,
`12: 0.434`, `20: 0.394`, `32: 0.378`, `48: 0.307`, `64: 0.252`) — and **1.72x** even
within `11+` alone.

### The finer grid is the obvious guess, and it is not the fix

Everything above says the bucket is too coarse, so the natural repair is Laya's **own**
mechanism with a finer grid: one *temperature* per option count. Fitted that way, the
temperatures come out at `k=4: 1.069`, `12: 1.415`, `20: 1.558`, `32: 1.554`, `48: 1.579`,
`64: 1.636` — a **1.53x** spread overall and only **1.16x** inside `choice:11+`. Measured
inside the bucket, n = 2,500 held out:

| form | params | ECE | Brier | bias |
|---|---:|---:|---:|---:|
| `p_top1` as published | 0 | 0.0515 | 0.0561 | +0.0504 |
| `T = 1.0` (no refit) | 0 | 0.0358 | 0.0507 | +0.0234 |
| one temperature, whole bucket | 1 | 0.0326 | 0.0531 | −0.0048 |
| **one temperature per option count** | **6** | **0.0338** | 0.0530 | −0.0049 |
| **per-k slope + one shared offset** | **7** | **0.0147** | 0.0467 | +0.0101 |
| escalation head | 5,543 | 0.0155 | 0.0476 | +0.0117 |

Six temperatures buy **nothing** over one (`0.0338` vs `0.0326`, marginally worse). The
reason is structural, not statistical: a temperature is a slope *through the origin* on
the logit scale, and this data wants an offset of **−0.234**. Forcing the origin
compresses every fitted slope toward 1, which is exactly why the temperature spread
(`1.53x`) looks so much smaller than the slope spread (`7.82x`). Add one shared offset
and the same six slopes are free to spread, cutting ECE a further **3.5x**.

So the correction needs seven numbers rather than six, and the seventh is not a
temperature. The consequence is strong: **no re-tuning of Laya's temperature grid can
repair this** — the mechanism being used to express the fix is the constraint.

The literal per-k temperature is not useless: it posts the best coverage @1% of any form
here (`0.773`). But it is worse on AURC (`0.0100` vs `0.0097`) and, inside the bucket, on
ECE (`0.0338` vs `0.0147`). Coverage at a target risk is a ranking quantity, and these
forms rank across option-count cells differently; ECE is what tells you whether the
number you publish means anything. It ships in `artifacts/real_big/temperatures.json` under
`negative_control`, because it is the guess everyone will make.

This is not a curve fit looking for a story: an independent diagnostic, run before the
fix was built, found the same thing. One shared recalibration of the bucket leaves a
per-`k` residual bias of `0.0310`, where a single map simulated on the same data
produces `0.0062 [0.0023, 0.0118]` — `p = 0.005`. The deployable form above (one slope per
`k`, one shared offset, so a single threshold still means the same thing everywhere) cuts
that to `0.0052`. See `artifacts/analysis_big.txt`.

## 5. The field Laya actually documents is worse than `p_top1`

Everything so far has been about `p_top1`, which you only get by reading the logits.
The field Laya documents is `confidence`, and `confidence` is **not** `p_top1` — it is
normalised entropy, `1 − H(p) / log k`. Auditing the documented field separately is not a
formality; it is a different curve with a worse defect, and it is the one users will call.

Held-out rows: 2,500 decisions, `choice:11+` only, 500 per option count. Every rescaling
below is fitted on the train split and scored on rows it never saw.

| k | accuracy | mean `confidence` | bias | ECE |
|---:|---:|---:|---:|---:|
| 12 | 0.984 | 0.9932 | +0.0092 | 0.0126 |
| 20 | 0.974 | 0.9910 | +0.0170 | 0.0186 |
| 32 | 0.960 | 0.9920 | +0.0320 | 0.0355 |
| 48 | 0.920 | 0.9803 | +0.0603 | 0.0659 |
| **64** | **0.784** | **0.9395** | **+0.1555** | **0.1555** |

A **16.9x** bias ramp, monotone, inside one temperature cell. And bias tracks ECE closely at
every `k` — equal at `k=64` (`0.1555`), within about `0.003` below it — so the error is
predominantly a **level** shift rather than a shape error. That is the signature a single
scalar rescale can repair, and it does:

| pool | pooled accuracy | pooled mean | pooled bias | max abs bias |
|---|---:|---:|---:|---:|
| as shipped (`T = 0.5`) | 0.9244 | 0.9792 | +0.0548 | 0.1555 |
| `T = 1.0` (0 floats, no refit) | 0.9244 | 0.9518 | +0.0274 | 0.0845 |
| `T = 1.592` (1 fitted float) | 0.9244 | 0.9321 | +0.0077 | **0.0393** |

Two independent objectives converge on the same value — `1.583` from minimising top-1
log-loss, `1.592` from matching mean confidence to accuracy — and one float cuts the worst
bias by **4.0x**. Worth repeating against the shipped constant: the checkpoint stores
`0.1006`, the loader refuses it, and the replacement `0.5` is roughly a third of what the
field wants.

What one float does not fix:

* **ECE at `k = 64`**: `0.1555` shipped → `0.1136` at `T = 1.0` → `0.1012` at `T = 1.592`.
  Better, still large. The level is repaired; the curve is not.
* **The grid is coarse on this field too.** Temperatures fitted per option count *to the
  `confidence` field* spread **2.07x** (`1.180 / 1.617 / 1.303 / 2.437 / 1.521`) — the
  same coarse-bucket defect, visible in the documented field rather than in `p_top1`.

### Rescaling repairs the level, not the risk

Each field gets its own threshold, fitted at 5% pooled risk on train, then measured
held-out:

| k | shipped: risk at its own threshold | rescaled at its own | rescaled at the *shipped* threshold |
|---:|---:|---:|---:|
| 12 | 0.0101 | 0.0102 | 0.0063 |
| 20 | 0.0082 | 0.0124 | 0.0085 |
| 32 | 0.0327 | 0.0328 | 0.0252 |
| 48 | 0.0665 | 0.0681 | 0.0507 |
| 64 | 0.1487 | 0.1366 | 0.0840 |

Spread inside `choice:11+`: **18.2x shipped → 13.3x rescaled**. Accuracy itself runs
`0.784 → 0.984` across those same counts, so no confidence scale can equalise risk
across them at any threshold. Rescaling fixes the field's *level*; it cannot fix the
accuracy decline. Reporting those as one defect would overstate the fix.

### Why one float is enough here and seven are not enough there

Worth making explicit, because the two sections could look contradictory. The `confidence`
field's defect is a **level** shift — bias tracks ECE at every `k` — and a level shift is
exactly what one scalar rescale removes. `p_top1`'s defect only *looks* the same until the
level is removed: what remains is a slope that differs per option count, and that needs one
slope per count plus the shared offset to express. The same asymmetry shows up in the
grids: fitted per option count, temperatures for `confidence` spread **2.07x** while
temperatures for `p_top1` spread only **1.53x**. The documented field is the one where the
coarse bucket bites hardest, and it is also the one a one-float fix can help most.

Reproduce with `USE_TF=0 HF_HUB_OFFLINE=1 python scripts/audit_confidence.py` (CPU,
reads `artifacts/sweep_big.npz`, writes `artifacts/confidence_audit.json`).

## 6. The learned head is a negative result — keep it out

`laya_escalate` also contains a 5,543-parameter MLP over 18 distribution-shape features
(`p_top2`, `margin`, `entropy_norm`, `top3_mass`, `effective_options`, and so on). It was
the original premise of this repository. The data does not support it:

* It **loses** to the seven floats on AURC (`0.0106` vs `0.0097`), on coverage @1%
  (`0.701` vs `0.737`), and on Brier (`0.0413` vs `0.0407`).
* It **ties** on coverage @2% (`0.891` vs `0.888`) and takes ECE by `0.0003`
  (`0.0123` vs `0.0126`). On the `>20` bucket it is a wash: ECE `0.0236` vs `0.0260`,
  bias `+0.0166` vs `+0.0168`.
* Against Laya's own `confidence` its AURC edge is only `1.30x` (`0.0138` vs
  `0.0106`) on held-out rows, while the seven-float correction gets `1.42x`
  (`0.0138` vs `0.0097`) against the same baseline.

The honest summary is that the head is **not distinguishable from a per-option-count
temperature, and is worse on the metrics that decide whether you can gate on it.** The
correct functional form beat the larger model class. The code stays in the repository
because that is the finding, not because it is the fix.

## 7. Why it is cardinality, and specifically not token starvation

The obvious mechanism story is that at high `k` each option gets less of the shared
`head_max_len = 512` header budget, so options get truncated and the task becomes
impossible. The diagnostic column in the sweep reports `tokens/option` as
`(512 − 16) / k`, which prints `7.8` at `k = 64` and reads like starvation.

**That column is arithmetic, not measurement.** `scripts/measure_option_tokens.py`
replays the collector's exact option draws and runs Laya's real
`build_sequence` with the real tokenizer:

| k | real option tokens (all options) | budget | per option | truncated? |
|---:|---:|---:|---:|---|
| 4 | 46 | 496 | 11.44 | no |
| 12 | 90 | 496 | 7.46 | no |
| 20 | 149 | 496 | 7.45 | no |
| 32 | 238 | 496 | 7.43 | no |
| 48 | 357 | 496 | 7.43 | no |
| 64 | 476 | 496 | 7.44 | **yes — 100% of questions** |

Per-option cost is essentially **flat in `k`** (7.4 tokens), because it is the same
distribution of short snake_case labels. Nothing is truncated at all for `k <= 48`,
which is where accuracy has already fallen `0.990 → 0.904`. Only `k = 64` overflows the
header budget, and even there the cut is partial: a per-option cap of 7 applies to 100%
of questions and drops **27.7%** of option tokens, while the instruction header and the
customer message survive intact (state truncated on 0% of questions).

So the collapse is **difficulty discriminating among many similar short labels**, not the
model running out of room to see them. This matters for the fix: a token-budget
explanation would call for a shorter option rendering, and that would not address it.

## 8. The part no recalibration of `p_top1` reaches: confusability at fixed `k`

Holding `k` fixed removes every handle a temperature could grab. Two sweeps at
**`k = 32` only**, same seed, same 2,000 items, same `head_max_len`, same applied bucket
temperature — differing *only* in which distractors were offered:

| condition | mean distractor name similarity | accuracy | mean `p_top1` | bias | ECE | NLL |
|---|---:|---:|---:|---:|---:|---:|
| `random` (uniform draw) | 0.009 | 0.943 | 0.985 | +0.0422 | 0.0443 | 0.4380 |
| `similar` (most confusable names) | 0.043 | 0.896 | 0.976 | +0.0335 | **0.0806** | **0.7408** |

206 of 2,000 decisions flipped. Raising confusability alone — with `T = 0.5` applied to
both — costs **4.7 points of accuracy and nearly doubles ECE and NLL**.

Pooling both conditions (4,000 decisions at one `k`) and asking whether a single
recalibration can serve them:

| confusability quartile | mean similarity | n | accuracy | ideal slope | residual bias under one map | ECE under one map |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.002 | 1000 | 0.953 | 0.320 | −0.0178 | 0.0222 |
| 2 | 0.009 | 1000 | 0.952 | 0.346 | −0.0123 | 0.0174 |
| 3 | 0.025 | 997 | 0.929 | 0.287 | −0.0008 | 0.0186 |
| 4 | 0.069 | 1003 | 0.844 | 0.251 | **+0.0308** | 0.0454 |

The residual is **monotone in confusability and changes sign** — one shared map is
overconfident on the easy quartile and underconfident on the hard one. Worst-case
residual `0.0308` against a null of `0.0082 [0.0025, 0.0185]`, `p = 0.005`; ideal slopes
spread `1.41x`, `p = 0.005`. Grouped by condition instead of quartile: residual
`−0.0091 / +0.0091`, `p = 0.010`.

**So a per-`k` temperature is necessary but not sufficient.** It fixes the `k`-driven
part, which is the dominant part (`7.8x` in bias versus this `~1.4x`). The remainder
needs a signal the temperature cannot see — which is what the head's shape features
would be for, and the reason that remains an open, honest avenue rather than a closed one.

### Ranking: where the head does and does not win

Worth stating because it is easy to conflate. `p_top1` scores **0.000** coverage at 1%
target risk because 79% of `choice:11+` decisions sit on exactly `1.0000` — a tie block no
threshold can split. That is what makes the head look strong against it (`0.701`).

But that is a property of the *unrecalibrated* field, not of the head. Once any
per-option-count recalibration is fitted, the tie block resolves into a usable ordering on
the same held-out rows: the seven-float affine form reaches **0.737** and one temperature
per option count reaches **0.773** — both **above** the head's `0.701`. The head's coverage
edge exists only against `p_top1` as published, the weakest baseline in the table, and on
AURC it loses to both (`0.0106` vs `0.0097` and `0.0100`).

So it is granularity *across* option counts, not extra features, that buys the coverage
here. The head's edge is real in exactly one place: **within** a single option count, where
a temperature is one map and therefore cannot reorder anything. On the pooled fixed-`k = 32`
data, where 79% of decisions share a published `p_top1 = 1.0000`, the best temperature
achieves **0.000** coverage at 1% target risk — no threshold can isolate a low-risk 1% out
of a 79% block of identical numbers — while the head reaches **0.316**, because its shape
features see something top-1 cannot. Its overall ranking is still worse there (AURC
`0.0216` vs `0.0180`). Ties induced by a clamp are the special case, not the rule.

## 9. How much data the claim needs — a cautionary pair

Both sweeps are committed, and they disagree:

| sweep | decisions | per-`k` slope spread | null `[lo, hi]` | p |
|---|---:|---:|---|---:|
| `sweep_small.npz` (200 items) | 1,000 | 1.86x | `[1.09, 2.29]` | **0.104 — not significant** |
| `sweep_big.npz` (2,000 items) | 12,000 | 1.58x | `[1.04, 1.17]` | **0.005 — significant** |

The *effect* got **smaller** (1.86x → 1.58x). The *null* got much tighter (upper bound
2.29 → 1.17). The first sweep's "not distinguishable from noise" was a shortage of data
and nothing else — and this repository published it as a conclusion before the second
sweep overturned it. `worst_abs_bias_shared` was significant at **both** sizes
(`p = 0.005`), so what `n = 200` specifically could not resolve was the slope. Compare
`sweep_small_analysis.json` against `sweep_big_analysis.json` to watch it happen.

## 10. Guarantees are sample-size dependent too

`fit_conformal_policy` + `evaluate` report the threshold that actually deploys, measured
on rows it was **not** fitted on:

| calibration rows | threshold | held-out coverage | measured risk | bound at `alpha = 0.010` |
|---:|---:|---:|---:|---|
| 250 | 0.8544 | 0.920 | **0.0240** | **VIOLATED, 2.4x over** |
| 3000 | 0.9748 | 0.637 | **0.0063** | holds |

`evaluate()` prints the threshold's source and an explicit note when the measured risk
exceeds the bound rather than quietly reporting the coverage. The 250-row violation is
why that note exists.

## 11. Two more things worth knowing before you deploy this

* **`act_probability` is degenerate on this corpus.** Laya's own act/escalate head returns
  exactly `1.0000` on all 12,000 sweep decisions and all 3,000 held-out rows. It cannot
  gate anything at any threshold — including the 20% of 64-option decisions that are
  wrong. AURC `0.0222`, the worst of every signal measured.
* **One global confidence gate distributes risk very unevenly.** A single `p_top1`
  threshold at 5% pooled selective risk yields per-`k` risk from **0.36%** at `k=4` to
  **15.0%** at `k=64` — a **42x** spread. On `confidence` it is **136.6x**. The pooled
  guarantee is real and it is a marginal statement; it does not cover the regime you
  happen to be serving.

## 12. Reproduce

```bash
USE_TF=0 python -m pytest -q          # 99 tests, no GPU, no downloads
```

CPU-only, reading the committed artifacts:

```bash
USE_TF=0 python scripts/analyze_sweep.py --sweep artifacts/sweep_big.npz \
    --compare artifacts/sweep_big_t1.npz --compare-label "T=1.0"
USE_TF=0 python scripts/train_escalate.py --records artifacts/sweep_big.npz --out artifacts/real_big
USE_TF=0 python scripts/audit_confidence.py
USE_TF=0 python scripts/analyze_sweep.py --sweep artifacts/sweep_k32_random.npz \
    --compare artifacts/sweep_k32_similar.npz --compare-label "similar distractors"
USE_TF=0 HF_HUB_OFFLINE=1 python scripts/measure_option_tokens.py --items 2000 \
    --k 4 12 20 32 48 64 --out artifacts/option_tokens.json
```

The collection commands, the environment gotchas, and a description of every committed
file are in [`artifacts/README.md`](artifacts/README.md).

## 13. Limitations

* The `confidence` audit in §5 covers `choice:11+` only (option counts 12–64) over 2,500
  held-out decisions. `k = 4` is a different temperature cell and is excluded there on
  purpose, which is why its risk spread (18.2x) is smaller than the whole-sweep figure in
  §11 (136.6x). They measure different populations, not disagreeing ones.

* One corpus. `clinc_oos/plus` has 150 short snake_case intent names. Option *text*
  length, option *count* and distractor confusability are therefore not independently
  varied across the whole design; §8 varies confusability at one `k`, and §7 shows the
  token cost stays flat, but a corpus with long option descriptions would exercise the
  header budget differently.
* One checkpoint (English) and one runtime (bf16, CUDA, batch size 1). The clamp is a
  property of the shipped config, so the `choice:11+` findings should transfer to the
  multilingual and typed-decisions checkpoints; the fitted floats are fit to this corpus
  and should be refit before being trusted elsewhere.
* The distractor-similarity ranking is over intent **names**, a cheap proxy for
  confusability. It moves the measured mean by 4.8x and the effect is significant, so it
  is doing something real, but it is not the same as semantic hard negatives built from
  utterance embeddings. The per-row achieved similarity is recorded in the artifacts
  precisely so a stronger selector can be substituted and the analysis re-run.
* The head's `cov @1%` advantage in the saturated fixed-`k` regime is measured on 1,000
  held-out rows, so 0.316 is roughly 316 decisions. Treat it as an indication.

## License

Apache-2.0, matching the model this work measures. See [`LICENSE`](LICENSE).
