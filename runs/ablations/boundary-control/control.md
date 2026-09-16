# Boundary-shuffled Lumber control

**Scope:** 20 QMSum validation meetings, 142 questions, Lumber target 1000 pseudo-tokens, 10 random partitions per meeting.

Each control permutes Lumber's exact turns-per-chunk distribution across legal turn boundaries. From random permutations, the closest word-size distribution is retained; semantic boundary locations are not used.

## Geometry check

| Chunks preserved | Lumber mean/median words | Control mean/median words | Lumber/control mean turns | Lumber boundaries retained |
|---:|---:|---:|---:|---:|
| True | 277.4/243.0 | 277.4/233.0 | 17.1/17.1 | 12.8% |

## Retrieval comparison

Values are meeting-macro averages. Differences are actual Lumber minus the mean shuffled control; intervals are paired 95% bootstrap intervals across meetings.

| Retriever | Budget | Lumber P/R/F1/Z | Control P/R/F1/Z | dRecall [95% CI] | dF1 [95% CI] | dMRR [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| dense | 512 | 0.386/0.332/0.315/35.8% | 0.344/0.302/0.279/35.2% | +0.030 [-0.007, +0.069] | +0.036 [+0.005, +0.068] | -0.029 [-0.066, +0.006] |
| dense | 1024 | 0.326/0.521/0.344/21.5% | 0.298/0.496/0.318/19.5% | +0.025 [-0.013, +0.062] | +0.027 [+0.003, +0.049] | -0.029 [-0.066, +0.006] |
| dense | 2048 | 0.229/0.673/0.293/11.9% | 0.224/0.663/0.286/8.7% | +0.010 [-0.028, +0.047] | +0.007 [-0.006, +0.021] | -0.029 [-0.066, +0.006] |
| bm25 | 512 | 0.329/0.293/0.274/42.0% | 0.276/0.248/0.230/44.2% | +0.045 [+0.003, +0.086] | +0.045 [+0.009, +0.080] | -0.004 [-0.033, +0.025] |
| bm25 | 1024 | 0.261/0.471/0.291/20.4% | 0.250/0.436/0.272/23.4% | +0.035 [-0.017, +0.086] | +0.019 [-0.008, +0.047] | -0.004 [-0.033, +0.025] |
| bm25 | 2048 | 0.207/0.620/0.267/12.5% | 0.198/0.596/0.255/12.1% | +0.025 [-0.013, +0.059] | +0.012 [+0.001, +0.022] | -0.004 [-0.033, +0.025] |
| hybrid | 512 | 0.349/0.305/0.289/37.7% | 0.336/0.300/0.275/36.9% | +0.005 [-0.033, +0.043] | +0.013 [-0.017, +0.042] | -0.030 [-0.057, -0.003] |
| hybrid | 1024 | 0.306/0.516/0.329/21.6% | 0.289/0.485/0.309/18.9% | +0.031 [-0.009, +0.070] | +0.019 [-0.001, +0.039] | -0.030 [-0.057, -0.003] |
| hybrid | 2048 | 0.226/0.655/0.289/11.6% | 0.218/0.646/0.279/9.4% | +0.009 [-0.034, +0.052] | +0.010 [-0.004, +0.025] | -0.030 [-0.057, -0.003] |

## Interpretation rule

Actual Lumber provides direct evidence for semantic boundary placement when its paired recall or F1 interval lies above zero. MRR remains secondary because overlap probability is size-sensitive, even though geometry is matched here.

This is a retrieval-only secondary ablation; no answers are generated.
