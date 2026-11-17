## Exp1 identifiable z-MAE

| arm | k=0 (n=20) | k=4 (n=20) | k=12 (n=20) | k=24 (n=20) |
|---|---|---|---|---|
| V-JEPA prior, compass 2-D | 0.243 | 0.259 | 0.221 | 0.185 |
| GT-law mean (priv.), compass 2-D | 0.629 (Δ +0.386 [+0.203, +0.544]) | 0.569 (Δ +0.310 [+0.071, +0.506]) | 0.313 (Δ +0.093 [-0.082, +0.224]) | 0.205 (Δ +0.020 [-0.081, +0.142]) |
| best-of-3 law means, compass 2-D | - | 0.698 (Δ +0.439 [+0.242, +0.560]) | 0.517 (Δ +0.296 [+0.054, +0.491]) | 0.445 (Δ +0.260 [+0.081, +0.418]) |
| V-JEPA prior, CMA-ES (s0=0.5) | 0.243 (Δ +0.000 [+0.000, +0.000]) | 0.202 (Δ -0.057 [-0.131, -0.004]) | 0.209 (Δ -0.011 [-0.123, +0.063]) | 0.154 (Δ -0.031 [-0.092, +0.031]) |
| GT-law mean (priv.), CMA-ES (s0=1) | 0.629 (Δ +0.386 [+0.203, +0.544]) | 0.469 (Δ +0.210 [+0.034, +0.354]) | 0.395 (Δ +0.174 [+0.017, +0.311]) | 0.327 (Δ +0.142 [+0.050, +0.230]) |
| best-of-3 law means, CMA-ES (s0=1) | - | 0.735 (Δ +0.476 [+0.322, +0.658]) | 0.566 (Δ +0.346 [+0.134, +0.503]) | 0.533 (Δ +0.348 [+0.111, +0.542]) |

## Exp1 sims-to-target

target T = 0.205 (n = 20)

| arm | sims to T | 95% CI (censored at 25) | P(reached) |
|---|---|---|---|
| V-JEPA prior, compass 2-D | 16.2 | [0.0, 25.0] | 0.70 |
| GT-law mean (priv.), compass 2-D | 24.0 | [6.6, 24.0] | 1.00 |
| best-of-3 law means, compass 2-D | >24 | [11.0, 25.0] | 0.03 |
| V-JEPA prior, CMA-ES (s0=0.5) | 3.0 | [0.0, 25.0] | 0.92 |
| GT-law mean (priv.), CMA-ES (s0=1) | >24 | [12.7, 25.0] | 0.06 |
| best-of-3 law means, CMA-ES (s0=1) | >24 | [25.0, 25.0] | 0.02 |

## Exp2 identifiable z-MAE

| arm | k=0 (n=20) | k=4 (n=20) | k=12 (n=20) | k=24 (n=20) |
|---|---|---|---|---|
| V-JEPA prior, compass 2-D | 0.243 | 0.259 | 0.221 | 0.185 |
| GT-law mean (priv.), compass 2-D | 0.629 (Δ +0.386 [+0.203, +0.544]) | 0.569 (Δ +0.310 [+0.071, +0.506]) | 0.313 (Δ +0.093 [-0.082, +0.224]) | 0.205 (Δ +0.020 [-0.081, +0.142]) |
| V-JEPA prior, compass 4-D (unfrozen h, nu) | 0.243 (Δ +0.000 [+0.000, +0.000]) | 0.259 (Δ +0.000 [+0.000, +0.000]) | 0.231 (Δ +0.011 [-0.014, +0.050]) | 0.192 (Δ +0.007 [-0.024, +0.035]) |
| GT-law mean (priv.), compass 4-D | 0.629 (Δ +0.386 [+0.203, +0.544]) | 0.564 (Δ +0.305 [+0.054, +0.503]) | 0.338 (Δ +0.117 [-0.077, +0.258]) | 0.294 (Δ +0.109 [-0.030, +0.213]) |

## Exp2 sims-to-target

target T = 0.205 (n = 20)

| arm | sims to T | 95% CI (censored at 25) | P(reached) |
|---|---|---|---|
| V-JEPA prior, compass 2-D | 16.2 | [0.0, 25.0] | 0.70 |
| V-JEPA prior, compass 4-D (unfrozen h, nu) | 22.3 | [0.0, 25.0] | 0.67 |
| GT-law mean (priv.), compass 2-D | 24.0 | [6.6, 24.0] | 1.00 |
| GT-law mean (priv.), compass 4-D | >24 | [8.9, 25.0] | 0.05 |

## Exp2 nuisance / fit / replay

| arm | k | n | ident z-MAE | abs hardening err | abs nu err | observed-fit IoU | replay IoU (n) |
|---|---|---|---|---|---|---|---|
| V-JEPA prior, compass 2-D | 0 | 20 | 0.243 | 0.083 (n=8) | 0.087 | - | - (0) |
| V-JEPA prior, compass 4-D (unfrozen h, nu) | 0 | 20 | 0.243 | 0.083 (n=8) | 0.087 | - | - (0) |
| GT-law mean (priv.), compass 2-D | 0 | 20 | 0.629 | 0.094 (n=12) | 0.088 | - | - (0) |
| GT-law mean (priv.), compass 4-D | 0 | 20 | 0.629 | 0.094 (n=12) | 0.088 | - | - (0) |
| V-JEPA prior, compass 2-D | 4 | 20 | 0.259 | 0.083 (n=8) | 0.087 | 0.802 | - (0) |
| V-JEPA prior, compass 4-D (unfrozen h, nu) | 4 | 20 | 0.259 | 0.083 (n=8) | 0.085 | 0.802 | - (0) |
| GT-law mean (priv.), compass 2-D | 4 | 20 | 0.569 | 0.094 (n=12) | 0.088 | 0.629 | - (0) |
| GT-law mean (priv.), compass 4-D | 4 | 20 | 0.564 | 0.094 (n=12) | 0.084 | 0.631 | - (0) |
| V-JEPA prior, compass 2-D | 12 | 20 | 0.221 | 0.083 (n=8) | 0.087 | 0.852 | 0.805 (18) |
| V-JEPA prior, compass 4-D (unfrozen h, nu) | 12 | 20 | 0.231 | 0.099 (n=8) | 0.080 | 0.841 | 0.798 (18) |
| GT-law mean (priv.), compass 2-D | 12 | 20 | 0.313 | 0.094 (n=12) | 0.088 | 0.861 | 0.751 (18) |
| GT-law mean (priv.), compass 4-D | 12 | 20 | 0.338 | 0.110 (n=12) | 0.070 | 0.854 | 0.713 (18) |
| V-JEPA prior, compass 2-D | 24 | 20 | 0.185 | 0.083 (n=8) | 0.087 | 0.875 | 0.826 (18) |
| V-JEPA prior, compass 4-D (unfrozen h, nu) | 24 | 20 | 0.192 | 0.150 (n=8) | 0.075 | 0.866 | 0.810 (18) |
| GT-law mean (priv.), compass 2-D | 24 | 20 | 0.205 | 0.094 (n=12) | 0.088 | 0.893 | 0.822 (18) |
| GT-law mean (priv.), compass 4-D | 24 | 20 | 0.294 | 0.107 (n=12) | 0.099 | 0.877 | 0.754 (18) |

## Coverage / failures

| arm | samples | with k=24 | sims (reused) | failed sims | samples with failures |
|---|---|---|---|---|---|
| V-JEPA prior, compass 2-D | 20 | 20 | 330 (0) | 27 | [113, 231] |
| GT-law mean (priv.), compass 2-D | 20 | 20 | 382 (0) | 17 | [113, 231] |
| best-of-3 law means, compass 2-D | 20 | 20 | 343 (0) | 26 | [113, 231] |
| V-JEPA prior, CMA-ES (s0=0.5) | 20 | 20 | 480 (19) | 24 | [113, 231] |
| GT-law mean (priv.), CMA-ES (s0=1) | 20 | 20 | 480 (18) | 26 | [113, 231] |
| best-of-3 law means, CMA-ES (s0=1) | 20 | 20 | 480 (55) | 33 | [113, 231] |
| V-JEPA prior, compass 4-D (unfrozen h, nu) | 20 | 20 | 477 (142) | 35 | [113, 231] |
| GT-law mean (priv.), compass 4-D | 20 | 20 | 477 (168) | 30 | [113, 202, 231] |
