# Artifacts

Every published number is derived from a file in this directory. They are
committed on purpose (about 2 MB total): it means the claims in the top-level
README can be re-derived on CPU, with no GPU and no model download, from
commands that only read these files.

Two things are recorded here rather than asserted: the corpus the sweeps ran on
is `clinc_oos/plus` test (150 in-scope intents, one out-of-scope class excluded
from the option pool), and every sweep holds `head_max_len` fixed at 512 so
cardinality is never confounded with the header budget setting.

## Canonical

| file | what it is |
|---|---|
| `sweep_big.npz` | 12,000 decisions: 2,000 items x k in {4,12,20,32,48,64}, shipped (clamped) temperatures. The primary evidence. |
| `sweep_big.meta.json` | Its configuration: dataset, seed, `head_max_len`, and the temperatures actually applied vs. the raw ones the checkpoint ships. |
| `sweep_big_t1.npz` | The `T = 1.0` ablation over the identical 12,000 items, for isolating the clamp from the cardinality effect. |
| `sweep_big_analysis.json` | Full diagnostic output for the pair above, including the null-calibrated slope-spread and bias statistics. |
| `analysis_big.txt` | Human-readable stdout of that same analysis. |
| `real_big/` | Trained 5,543-parameter head bundle plus `report.json`, from the canonical `train_escalate.py` run. |
| `real_big/temperatures.json` | **The fix, as a loadable file.** Seven floats: one logistic slope per option count on `logit(p_top1)`, plus a single shared offset. Also carries the per-option-count *temperature* grid under `negative_control` — the form that does **not** work, kept because it is the obvious guess. |
| `head_report.txt` | Stdout of that run: the held-out scorer comparison, the per-bucket table, and the `choice:11+` comparison showing that a finer temperature grid is not the fix. |
| `option_tokens.json` | Real rendered option-token counts per `k`, measured with the Laya tokenizer. Settles the mechanism question. |

## The documented `confidence` field

| file | what it is |
|---|---|
| `confidence_audit.json` | Held-out audit of the field Laya actually documents. `confidence` is normalised entropy, **not** `p_top1`, and its bias ramps **16.9x** across `choice:11+` (`+0.0092` at k=12 to `+0.1555` at k=64). Records the field as shipped (`T = 0.5`), at `T = 1.0`, and under one fitted global `T = 1.592`; the per-option-count temperatures fitted to this field; and per-`k` risk at each threshold. |

`confidence_audit.json` is the one artifact whose comparison is a *field* rather than a
scorer: everything else audits `p_top1`, which a user only sees by reading the logits.
All temperatures in it are fitted on the train split and scored on held-out rows, and
both thresholds are fitted at 5% pooled risk on train, so neither side of the comparison
gets an advantage.

## Within-`k` experiment (fixed cardinality)

| file | what it is |
|---|---|
| `sweep_k32_random.npz` | 2,000 decisions at **k = 32 only**, distractors drawn uniformly. The low-confusability condition (mean pairwise name similarity 0.009). |
| `sweep_k32_similar.npz` | The same 2,000 items at the same `k = 32`, distractors chosen as the most confusable intent names. The high-confusability condition (0.043). |
| `sweep_k32_pooled.npz` | The two conditions stacked into one 4,000-decision file, with a `condition` column. Input to the head run. |
| `within_k_analysis.json` | The within-`k` adequacy analysis: can one recalibration serve every group? |
| `real_k32_pooled/` | Head trained and evaluated on the pooled fixed-`k` data. |

`sweep_k32_random` / `sweep_k32_similar` are paired on `item_index`: same items,
same option count, same `head_max_len`, same applied bucket temperature. The
only thing that differs is which distractors were offered.

## Kept deliberately: the small sweep

| file | what it is |
|---|---|
| `sweep_small.npz` | 1,000 decisions: 200 items x k in {4,12,20,32,64}. |
| `sweep_small_t1.npz` | Its `T = 1.0` ablation. |
| `sweep_small_analysis.json` | Its analysis. |

This one is **superseded as evidence but retained as a cautionary record**, because
it is the reason the project's first conclusion was wrong. On 1,000 decisions the
per-`k` slope spread is 1.86x against a null of [1.09, 2.29] -> p = 0.104, i.e.
"not distinguishable from noise", and the obvious reading is that a per-`k`
temperature is not justified. At 12,000 decisions the same statistic is 1.58x
against a null of [1.04, 1.17] -> p = 0.005.

Note which number moved. The *effect* got **smaller** (1.86x -> 1.58x); the
*null* got much tighter (upper bound 2.29 -> 1.17). The false negative was
entirely a shortage of data, not an absent effect. `worst_abs_bias_shared` was
significant at both sizes (p = 0.005), so the bias evidence was already there at
200 items -- it was specifically the slope evidence that n = 200 could not
resolve. Compare the two analysis JSONs and you can watch it happen.

## Reproduce

```bash
# 12,000 real decisions, ~9 min on an 8 GB RTX 4060 (bf16, batch size 1)
USE_TF=0 python scripts/collect_sweep.py --dataset clinc_oos --subset plus --split test \
    --items 2000 --k 4 12 20 32 48 64 --head-max-len 512 --seed 0 \
    --out artifacts/sweep_big.npz

# the T = 1.0 ablation, same items and order
USE_TF=0 python scripts/collect_sweep.py --dataset clinc_oos --subset plus --split test \
    --items 2000 --k 4 12 20 32 48 64 --head-max-len 512 --seed 0 \
    --temperature-override 1.0 --out artifacts/sweep_big_t1.npz

# diagnostic + null-calibrated significance + clamp ablation
USE_TF=0 python scripts/analyze_sweep.py --sweep artifacts/sweep_big.npz \
    --compare artifacts/sweep_big_t1.npz --compare-label "T=1.0" \
    --out artifacts/sweep_big_analysis.json

# head + threshold + baseline comparison (CPU, ~1 min)
USE_TF=0 python scripts/train_escalate.py --records artifacts/sweep_big.npz --out artifacts/real_big

# the documented confidence field: shipped vs T=1.0 vs one fitted float (CPU, offline)
USE_TF=0 HF_HUB_OFFLINE=1 python scripts/audit_confidence.py

# mechanism: real option-token counts and whether anything is truncated (CPU, offline)
USE_TF=0 HF_HUB_OFFLINE=1 python scripts/measure_option_tokens.py --items 2000 \
    --k 4 12 20 32 48 64 --out artifacts/option_tokens.json

# within-k experiment at fixed k = 32 (2 x ~2 min on the same GPU)
USE_TF=0 python scripts/collect_sweep.py --dataset clinc_oos --subset plus --split test \
    --items 2000 --k 32 --head-max-len 512 --seed 0 --distractors random \
    --out artifacts/sweep_k32_random.npz
USE_TF=0 python scripts/collect_sweep.py --dataset clinc_oos --subset plus --split test \
    --items 2000 --k 32 --head-max-len 512 --seed 0 --distractors similar \
    --out artifacts/sweep_k32_similar.npz

# within-k analysis: does one recalibration serve every confusability level?
USE_TF=0 python scripts/analyze_sweep.py --sweep artifacts/sweep_k32_random.npz \
    --compare artifacts/sweep_k32_similar.npz --compare-label "similar distractors" \
    --out artifacts/within_k_analysis.json
```

`scripts/collect_sweep.py --audit-options` sizes the distractor manipulation
without loading a model, and `--show-similar N` prints the most confusable
neighbours of `N` random intents so you can check the selector is doing what it
claims.

## Environment gotchas

* `USE_TF=0`, or `transformers` can deadlock during model construction.
* One question per call. A 20-question batch exceeds 8 GB of VRAM and Laya
  silently falls back to CPU, which would contaminate both latency and throughput.
* Detached background runs do not survive across sessions; run synchronously.
* Python block-buffers stdout when redirected, so a long run written to a log
  looks stalled while it is working. Watch `nvidia-smi`, not the log.
* `clinc_oos` is parquet-hosted and works offline with `HF_HUB_OFFLINE=1`.
  `PolyAI/banking77` no longer loads under `datasets >= 4`, which is why the
  corpus is CLINC150.
