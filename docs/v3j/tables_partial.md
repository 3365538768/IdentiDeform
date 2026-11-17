## Exp1 identifiable z-MAE

| arm | k=0 (n=8) | k=4 (n=7) | k=12 (n=7) | k=24 (n=0) |
|---|---|---|---|---|
| V-JEPA prior, compass 2-D | 0.278 | 0.236 | 0.085 | - |
| GT-law mean (priv.), compass 2-D | 0.810 (Δ +0.532 [+0.140, +0.746]) | 0.343 (Δ +0.106 [-0.184, +0.460]) | 0.048 (Δ -0.037 [-0.051, -0.012]) | - |
| best-of-3 law means, compass 2-D | - | 0.661 (Δ +0.425 [+0.067, +0.760]) | 0.253 (Δ +0.168 [-0.057, +0.521]) | - |
| V-JEPA prior, CMA-ES (s0=0.5) | 0.278 (Δ +0.000 [+0.000, +0.000]) | 0.211 (Δ -0.025 [-0.178, +0.116]) | 0.145 (Δ +0.060 [-0.005, +0.118]) | - |
| GT-law mean (priv.), CMA-ES (s0=1) | 0.810 (Δ +0.532 [+0.140, +0.746]) | 0.386 (Δ +0.150 [-0.071, +0.440]) | 0.257 (Δ +0.172 [-0.022, +0.459]) | - |
| best-of-3 law means, CMA-ES (s0=1) | - | 0.749 (Δ +0.513 [+0.351, +0.696]) | 0.331 (Δ +0.246 [-0.026, +0.586]) | - |

## Exp1 (arms with budget 24)

| arm | k=0 (n=20) | k=4 (n=20) | k=12 (n=20) | k=24 (n=20) |
|---|---|---|---|---|
| V-JEPA prior, compass 2-D | 0.243 | 0.259 | 0.221 | 0.185 |
| GT-law mean (priv.), compass 2-D | 0.629 (Δ +0.386 [+0.203, +0.544]) | 0.569 (Δ +0.310 [+0.071, +0.506]) | 0.313 (Δ +0.093 [-0.082, +0.224]) | 0.205 (Δ +0.020 [-0.081, +0.142]) |
| best-of-3 law means, compass 2-D | - | 0.698 (Δ +0.439 [+0.242, +0.560]) | 0.517 (Δ +0.296 [+0.054, +0.491]) | 0.445 (Δ +0.260 [+0.081, +0.418]) |

## Exp1 sims-to-target

target T = 0.205 (n = 20)

| arm | sims to T | 95% CI (censored at 25) | P(reached) |
|---|---|---|---|
| V-JEPA prior, compass 2-D | 16.2 | [0.0, 25.0] | 0.70 |
| GT-law mean (priv.), compass 2-D | 24.0 | [6.6, 24.0] | 1.00 |
| best-of-3 law means, compass 2-D | >24 | [11.0, 25.0] | 0.03 |

## Exp2 identifiable z-MAE

| arm | k=0 (n=15) | k=4 (n=15) | k=12 (n=15) | k=24 (n=15) |
|---|---|---|---|---|
| V-JEPA prior, compass 2-D | 0.234 | 0.253 | 0.247 | 0.204 |
| GT-law mean (priv.), compass 2-D | 0.630 (Δ +0.395 [+0.183, +0.564]) | 0.617 (Δ +0.365 [+0.151, +0.569]) | 0.321 (Δ +0.074 [-0.129, +0.234]) | 0.189 (Δ -0.016 [-0.107, +0.106]) |
| V-JEPA prior, compass 4-D (unfrozen h, nu) | 0.234 (Δ +0.000 [+0.000, +0.000]) | 0.253 (Δ +0.000 [+0.000, +0.000]) | 0.267 (Δ +0.020 [-0.003, +0.066]) | 0.223 (Δ +0.019 [-0.007, +0.046]) |
| GT-law mean (priv.), compass 4-D | 0.630 (Δ +0.395 [+0.183, +0.564]) | 0.617 (Δ +0.365 [+0.151, +0.569]) | 0.349 (Δ +0.102 [-0.124, +0.272]) | 0.295 (Δ +0.091 [-0.065, +0.211]) |

## Exp2 sims-to-target

target T = 0.189 (n = 15)

| arm | sims to T | 95% CI (censored at 25) | P(reached) |
|---|---|---|---|
| V-JEPA prior, compass 2-D | >24 | [0.0, 25.0] | 0.53 |
| V-JEPA prior, compass 4-D (unfrozen h, nu) | >24 | [0.0, 25.0] | 0.45 |
| GT-law mean (priv.), compass 2-D | 24.0 | [8.9, 24.0] | 1.00 |
| GT-law mean (priv.), compass 4-D | >24 | [10.0, 25.0] | 0.03 |

## Exp2 nuisance / fit / replay

| arm | k | n | ident z-MAE | abs hardening err | abs nu err | observed-fit IoU | replay IoU (n) |
|---|---|---|---|---|---|---|---|
| V-JEPA prior, compass 2-D | 0 | 15 | 0.234 | 0.083 (n=8) | 0.082 | - | - (0) |
| V-JEPA prior, compass 4-D (unfrozen h, nu) | 0 | 15 | 0.234 | 0.083 (n=8) | 0.082 | - | - (0) |
| GT-law mean (priv.), compass 2-D | 0 | 15 | 0.630 | 0.098 (n=10) | 0.082 | - | - (0) |
| GT-law mean (priv.), compass 4-D | 0 | 15 | 0.630 | 0.098 (n=10) | 0.082 | - | - (0) |
| V-JEPA prior, compass 2-D | 4 | 15 | 0.253 | 0.083 (n=8) | 0.082 | 0.792 | - (0) |
| V-JEPA prior, compass 4-D (unfrozen h, nu) | 4 | 15 | 0.253 | 0.083 (n=8) | 0.079 | 0.792 | - (0) |
| GT-law mean (priv.), compass 2-D | 4 | 15 | 0.617 | 0.098 (n=10) | 0.082 | 0.586 | - (0) |
| GT-law mean (priv.), compass 4-D | 4 | 15 | 0.617 | 0.098 (n=10) | 0.082 | 0.586 | - (0) |
| V-JEPA prior, compass 2-D | 12 | 15 | 0.247 | 0.083 (n=8) | 0.082 | 0.836 | 0.823 (10) |
| V-JEPA prior, compass 4-D (unfrozen h, nu) | 12 | 15 | 0.267 | 0.099 (n=8) | 0.098 | 0.826 | 0.814 (10) |
| GT-law mean (priv.), compass 2-D | 12 | 15 | 0.321 | 0.098 (n=10) | 0.082 | 0.868 | 0.776 (10) |
| GT-law mean (priv.), compass 4-D | 12 | 15 | 0.349 | 0.108 (n=10) | 0.072 | 0.862 | 0.754 (10) |
| V-JEPA prior, compass 2-D | 24 | 15 | 0.204 | 0.083 (n=8) | 0.082 | 0.867 | 0.860 (10) |
| V-JEPA prior, compass 4-D (unfrozen h, nu) | 24 | 15 | 0.223 | 0.150 (n=8) | 0.087 | 0.850 | 0.839 (10) |
| GT-law mean (priv.), compass 2-D | 24 | 15 | 0.189 | 0.098 (n=10) | 0.082 | 0.904 | 0.841 (10) |
| GT-law mean (priv.), compass 4-D | 24 | 15 | 0.295 | 0.109 (n=10) | 0.107 | 0.883 | 0.783 (10) |

## Coverage / failures

| arm | samples | with k=24 | sims (reused) | failed sims | samples with failures |
|---|---|---|---|---|---|
| V-JEPA prior, compass 2-D | 20 | 20 | 330 (0) | 27 | [113, 231] |
| GT-law mean (priv.), compass 2-D | 20 | 20 | 382 (0) | 17 | [113, 231] |
| best-of-3 law means, compass 2-D | 20 | 20 | 343 (0) | 26 | [113, 231] |
| V-JEPA prior, CMA-ES (s0=0.5) | 8 | 0 | 96 (8) | 0 | [] |
| GT-law mean (priv.), CMA-ES (s0=1) | 8 | 0 | 96 (8) | 0 | [] |
| best-of-3 law means, CMA-ES (s0=1) | 7 | 0 | 84 (21) | 0 | [] |
| V-JEPA prior, compass 4-D (unfrozen h, nu) | 16 | 16 | 381 (128) | 35 | [113, 231] |
| GT-law mean (priv.), compass 4-D | 17 | 17 | 405 (156) | 30 | [113, 202, 231] |
