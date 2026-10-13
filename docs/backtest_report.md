# CPI forecast backtest — DOSM data, origins Jan 2018 – Jul 2026

**rel_mae** < 1.00 means the model beats the naive forecast ("next month = this month"). **mae_pp** = mean error of the cumulative % change, in percentage points.

`ets` and `ens_ets_direct` run on every 3rd origin (ETS is slow), so compare models with **rel_mae**, which is computed on matched forecasts; mae_pp/mase/dir_acc for those two cover a subset of origins.

`xgb_leaky_onestep` is scored the way train.py scored it (leaky features, one step). It is not a real forecast — it shows how much the leak flatters the reported metrics.

## All

**rel_mae**

| model | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|
| naive | 1.00 | 1.00 | 1.00 | 1.00 |
| snaive_growth | 1.60 | 1.49 | 1.33 | 1.18 |
| drift | 1.17 | 1.17 | 1.18 | 1.18 |
| ets | 1.06 | 1.01 | 0.99 | 0.96 |
| xgb_direct_growth | 1.04 | 0.97 | 0.95 | 0.99 |
| ens_ets_direct | 1.03 | 0.97 | 0.94 | 0.94 |
| xgb_leak_fixed | 2.28 | 1.65 | 1.49 | 1.37 |
| xgb_as_deployed | 2.23 | 1.37 | 1.22 | 1.11 |
| xgb_leaky_onestep | 1.81 | — | — | — |

**mae_pp**

| model | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|
| naive | 0.30 | 0.71 | 1.16 | 1.96 |
| snaive_growth | 0.49 | 1.06 | 1.54 | 2.32 |
| drift | 0.35 | 0.83 | 1.37 | 2.32 |
| ets | 0.25 | 0.74 | 1.16 | 1.88 |
| xgb_direct_growth | 0.31 | 0.69 | 1.10 | 1.93 |
| ens_ets_direct | 0.25 | 0.70 | 1.10 | 1.85 |
| xgb_leak_fixed | 0.69 | 1.18 | 1.73 | 2.69 |
| xgb_as_deployed | 0.68 | 0.98 | 1.41 | 2.18 |
| xgb_leaky_onestep | 0.55 | — | — | — |

**mase**

| model | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|
| naive | 1.04 | 2.57 | 4.33 | 7.50 |
| snaive_growth | 1.56 | 3.52 | 5.33 | 8.23 |
| drift | 1.23 | 2.93 | 4.88 | 8.23 |
| ets | 0.93 | 2.57 | 4.21 | 6.94 |
| xgb_direct_growth | 1.10 | 2.57 | 4.30 | 7.64 |
| ens_ets_direct | 0.90 | 2.49 | 4.14 | 7.05 |
| xgb_leak_fixed | 2.43 | 4.42 | 6.75 | 10.82 |
| xgb_as_deployed | 2.31 | 3.53 | 5.34 | 8.51 |
| xgb_leaky_onestep | 1.89 | — | — | — |

**dir_acc**

| model | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|
| naive | — | — | — | — |
| snaive_growth | 46% | 58% | 65% | 70% |
| drift | 63% | 67% | 68% | 70% |
| ets | 62% | 64% | 64% | 66% |
| xgb_direct_growth | 69% | 74% | 76% | 76% |
| ens_ets_direct | 70% | 74% | 76% | 77% |
| xgb_leak_fixed | 61% | 66% | 68% | 70% |
| xgb_as_deployed | 56% | 64% | 67% | 69% |
| xgb_leaky_onestep | 70% | — | — | — |

## Pre Covid

**rel_mae**

| model | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|
| naive | 1.00 | 1.00 | 1.00 | 1.00 |
| snaive_growth | 1.77 | 1.54 | 1.39 | 1.33 |
| drift | 1.19 | 1.19 | 1.26 | 1.33 |
| ets | 1.13 | 1.08 | 1.10 | 1.15 |
| xgb_direct_growth | 1.12 | 1.02 | 1.11 | 1.37 |
| ens_ets_direct | 1.07 | 1.03 | 1.07 | 1.22 |
| xgb_leak_fixed | 2.37 | 1.73 | 1.74 | 1.79 |
| xgb_as_deployed | 2.19 | 1.30 | 1.27 | 1.29 |
| xgb_leaky_onestep | 1.76 | — | — | — |

**mae_pp**

| model | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|
| naive | 0.32 | 0.88 | 1.31 | 1.86 |
| snaive_growth | 0.57 | 1.35 | 1.82 | 2.47 |
| drift | 0.39 | 1.05 | 1.64 | 2.47 |
| ets | 0.22 | 1.01 | 1.51 | 2.16 |
| xgb_direct_growth | 0.36 | 0.90 | 1.46 | 2.55 |
| ens_ets_direct | 0.21 | 0.96 | 1.46 | 2.30 |
| xgb_leak_fixed | 0.77 | 1.52 | 2.27 | 3.33 |
| xgb_as_deployed | 0.71 | 1.14 | 1.66 | 2.39 |
| xgb_leaky_onestep | 0.57 | — | — | — |

**mase**

| model | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|
| naive | 1.08 | 2.92 | 4.32 | 5.97 |
| snaive_growth | 1.77 | 4.33 | 6.04 | 8.44 |
| drift | 1.32 | 3.54 | 5.51 | 8.44 |
| ets | 0.69 | 3.16 | 4.84 | 6.70 |
| xgb_direct_growth | 1.25 | 3.04 | 4.99 | 8.66 |
| ens_ets_direct | 0.67 | 3.08 | 4.85 | 7.47 |
| xgb_leak_fixed | 2.66 | 5.40 | 8.22 | 12.34 |
| xgb_as_deployed | 2.42 | 3.89 | 5.71 | 8.34 |
| xgb_leaky_onestep | 1.96 | — | — | — |

**dir_acc**

| model | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|
| naive | — | — | — | — |
| snaive_growth | 48% | 55% | 55% | 58% |
| drift | 56% | 56% | 56% | 58% |
| ets | 68% | 63% | 62% | 67% |
| xgb_direct_growth | 65% | 66% | 69% | 69% |
| ens_ets_direct | 70% | 68% | 69% | 70% |
| xgb_leak_fixed | 59% | 62% | 62% | 62% |
| xgb_as_deployed | 57% | 62% | 63% | 62% |
| xgb_leaky_onestep | 71% | — | — | — |

## Covid

**rel_mae**

| model | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|
| naive | 1.00 | 1.00 | 1.00 | 1.00 |
| snaive_growth | 1.82 | 1.88 | 1.66 | 1.47 |
| drift | 1.28 | 1.39 | 1.48 | 1.47 |
| ets | 1.27 | 1.10 | 1.07 | 0.99 |
| xgb_direct_growth | 1.06 | 0.99 | 0.87 | 0.91 |
| ens_ets_direct | 1.19 | 1.00 | 0.93 | 0.92 |
| xgb_leak_fixed | 2.73 | 2.16 | 1.98 | 1.62 |
| xgb_as_deployed | 2.49 | 1.58 | 1.43 | 1.20 |
| xgb_leaky_onestep | 1.93 | — | — | — |

**mae_pp**

| model | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|
| naive | 0.35 | 0.74 | 1.16 | 2.12 |
| snaive_growth | 0.64 | 1.39 | 1.92 | 3.11 |
| drift | 0.45 | 1.03 | 1.71 | 3.11 |
| ets | 0.26 | 0.77 | 1.21 | 2.08 |
| xgb_direct_growth | 0.38 | 0.73 | 1.00 | 1.93 |
| ens_ets_direct | 0.25 | 0.70 | 1.05 | 1.94 |
| xgb_leak_fixed | 0.97 | 1.60 | 2.29 | 3.43 |
| xgb_as_deployed | 0.88 | 1.17 | 1.66 | 2.53 |
| xgb_leaky_onestep | 0.68 | — | — | — |

**mase**

| model | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|
| naive | 0.84 | 1.87 | 3.04 | 5.73 |
| snaive_growth | 1.38 | 3.00 | 4.20 | 7.22 |
| drift | 1.04 | 2.38 | 3.92 | 7.22 |
| ets | 0.78 | 1.90 | 3.10 | 5.36 |
| xgb_direct_growth | 0.94 | 2.15 | 3.28 | 6.55 |
| ens_ets_direct | 0.72 | 1.84 | 2.97 | 5.63 |
| xgb_leak_fixed | 2.95 | 5.52 | 8.32 | 12.56 |
| xgb_as_deployed | 2.36 | 3.50 | 5.19 | 7.93 |
| xgb_leaky_onestep | 1.96 | — | — | — |

**dir_acc**

| model | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|
| naive | — | — | — | — |
| snaive_growth | 41% | 53% | 62% | 64% |
| drift | 57% | 60% | 64% | 64% |
| ets | 56% | 63% | 64% | 65% |
| xgb_direct_growth | 68% | 76% | 80% | 77% |
| ens_ets_direct | 64% | 77% | 82% | 80% |
| xgb_leak_fixed | 57% | 61% | 64% | 68% |
| xgb_as_deployed | 55% | 62% | 68% | 73% |
| xgb_leaky_onestep | 67% | — | — | — |

## Post Covid

**rel_mae**

| model | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|
| naive | 1.00 | 1.00 | 1.00 | 1.00 |
| snaive_growth | 1.40 | 1.26 | 1.13 | 0.94 |
| drift | 1.10 | 1.04 | 1.00 | 0.94 |
| ets | 0.99 | 0.93 | 0.88 | 0.83 |
| xgb_direct_growth | 0.98 | 0.92 | 0.88 | 0.81 |
| ens_ets_direct | 0.97 | 0.90 | 0.86 | 0.79 |
| xgb_leak_fixed | 2.00 | 1.34 | 1.11 | 1.00 |
| xgb_as_deployed | 2.13 | 1.31 | 1.09 | 0.97 |
| xgb_leaky_onestep | 1.79 | — | — | — |

**mae_pp**

| model | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|
| naive | 0.27 | 0.62 | 1.08 | 1.94 |
| snaive_growth | 0.38 | 0.78 | 1.22 | 1.83 |
| drift | 0.30 | 0.65 | 1.08 | 1.83 |
| ets | 0.27 | 0.59 | 0.96 | 1.62 |
| xgb_direct_growth | 0.27 | 0.58 | 0.95 | 1.57 |
| ens_ets_direct | 0.26 | 0.57 | 0.93 | 1.55 |
| xgb_leak_fixed | 0.54 | 0.84 | 1.20 | 1.94 |
| xgb_as_deployed | 0.58 | 0.82 | 1.18 | 1.88 |
| xgb_leaky_onestep | 0.49 | — | — | — |

**mase**

| model | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|
| naive | 1.10 | 2.69 | 4.90 | 9.29 |
| snaive_growth | 1.54 | 3.34 | 5.45 | 8.61 |
| drift | 1.26 | 2.85 | 4.98 | 8.61 |
| ets | 1.09 | 2.52 | 4.33 | 7.83 |
| xgb_direct_growth | 1.10 | 2.51 | 4.40 | 7.59 |
| ens_ets_direct | 1.07 | 2.45 | 4.24 | 7.47 |
| xgb_leak_fixed | 2.11 | 3.47 | 5.30 | 9.06 |
| xgb_as_deployed | 2.23 | 3.36 | 5.21 | 8.91 |
| xgb_leaky_onestep | 1.82 | — | — | — |

**dir_acc**

| model | h=1 | h=3 | h=6 | h=12 |
|---|---:|---:|---:|---:|
| naive | — | — | — | — |
| snaive_growth | 47% | 62% | 70% | 79% |
| drift | 69% | 74% | 77% | 79% |
| ets | 61% | 64% | 64% | 65% |
| xgb_direct_growth | 71% | 76% | 79% | 80% |
| ens_ets_direct | 72% | 76% | 78% | 80% |
| xgb_leak_fixed | 63% | 69% | 73% | 76% |
| xgb_as_deployed | 55% | 65% | 68% | 71% |
| xgb_leaky_onestep | 70% | — | — | — |
