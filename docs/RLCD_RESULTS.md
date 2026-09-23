# RLCD-style browser decision results

All rows start from the same supervised Qwen3.5-0.8B JevForge checkpoint and
use the same frozen website-disjoint evaluation records. The RL rows use group
sampling, a Brier calibration term, and KL anchoring to that checkpoint.

| Metric | Supervised | RLCD-style | + 1 hard-negative view | + 2 hard-negative views |
|---|---:|---:|---:|---:|
| Test choice top-1 | 0.5787 | **0.5875** | 0.5863 | 0.5775 |
| OOD choice top-1 | 0.6373 | 0.6425 | **0.6477** | **0.6477** |
| Test choice positive mass | 0.4491 | 0.4869 | **0.4881** | 0.4859 |
| OOD choice positive mass | 0.4790 | 0.5182 | **0.5328** | 0.5304 |
| Test noul accuracy | 0.8263 | 0.8313 | 0.8275 | **0.8337** |
| OOD noul accuracy | 0.8601 | 0.8497 | 0.8679 | **0.8705** |
| Test noul Brier | 0.1283 | 0.1153 | 0.1147 | **0.1107** |
| OOD noul Brier | 0.1173 | 0.1087 | 0.1048 | **0.1038** |
| Test score MAE | **0.3674** | 0.3839 | 0.4232 | 0.3889 |
| OOD score MAE | 0.3898 | **0.3854** | 0.4285 | 0.3880 |

The plain RL update gives the best in-domain browser choice result while also
improving OOD choice. Two synthetic hard-negative views shift more probability
onto valid OOD actions and improve OOD top-1, but lose the in-domain choice
gain. One view retains almost all of the plain RL test gain while matching the
two-view OOD top-1 and producing the highest positive mass. It is the best
browser-focused compromise in this sweep, but its ordered score regression
means it should not replace the general checkpoint. This is evidence for a
mixture trade-off, not for scaling synthetic views without limit.

W&B runs:

- [Plain sampled RLCD-style run](https://wandb.ai/760243703-renmin-university-of-china/jevforge/runs/xpug990k)
- [Two-view browser augmentation run](https://wandb.ai/760243703-renmin-university-of-china/jevforge/runs/ejef9mpd)
- [One-view browser augmentation run](https://wandb.ai/760243703-renmin-university-of-china/jevforge/runs/wgve64v4)
