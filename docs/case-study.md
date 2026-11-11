# Case study: my model was 97% accurate, until I tested it properly

**Malaysia Personal Inflation Intelligence** forecasts consumer prices for 16 states × 13 spending categories. It turns those forecasts into a personal inflation rate for each user. This is the story of how I found that my headline metrics were wrong, proved it, and rebuilt the model so its claims hold up.

**TL;DR**
- I published **MAE 0.73 and 97% directional accuracy**.
- An honest backtest showed the deployed model was **2.2× less accurate than simply assuming prices stay the same** one month ahead, and worse than that baseline at every horizon.
- The redesigned model beats that baseline by 3–6% overall and by 10–21% since 2022. Its 80% forecast ranges contained the real outcome **81%** of the time in an out-of-sample test.
- Production now scores every published forecast against the real data as it arrives.

---

## 1. The claim

The original training script reported:

| Metric | Prophet baseline | XGBoost (deployed) |
|---|---:|---:|
| MAE | 5.549 | 0.733 |
| Directional accuracy | 0.75% | 96.98% |

A 97% hit rate on predicting the direction of inflation, against a baseline at under 1%, should have been a red flag. Inflation is hard to forecast. Numbers that good usually mean the test is broken, not that the model is brilliant.

## 2. What was wrong, from first principles

**The model could see the answer.** Two features, the 3- and 12-month moving averages of the CPI index, were computed over windows that *included the month being predicted*. So the target sat inside its own inputs. That's why the 3-month average came out as the "most important feature".

**The test didn't match the product.** Training scored one-step-ahead predictions built from real data. The app forecasts 3, 6 and 12 months ahead *recursively*, feeding each prediction back in as an input. It also computes the moving averages from past months only, the honest way, so the deployed model received inputs that meant something different from what it was trained on. No metric described what users actually saw.

**The direction metric compared the wrong rows.** The table was sorted by date across all 208 series, so "did the direction match?" compared neighbouring rows from *different* states and categories. The Prophet baseline was one model pooled across every series, so it predicted the same value for every row on a given date. That's why it scored 0.75%.

**The baseline was a straw man.** For a slow-moving series like CPI, the bar to beat is "next month = this month". The project never compared against it.

## 3. Building a test I could trust

I wrote a **rolling-origin backtest** (`ml/training/backtest.py`):

- **Origins:** for every month from January 2018 to July 2026, cut the data off at that month. Retrain every model using only what existed then.
- **Forecasts:** go 1–12 months ahead for all 208 series. The deployed model is run through the *exact* procedure the API uses.
- **Scoring:** compare with what actually happened, using error on the % change from the forecast month. The key number is error relative to the naive "no change" forecast, on matched forecasts (**below 1.0 beats naive**). Direction is measured *within* each series.
- **Self-test:** multiply every value after the origin by 50 and assert that no forecast changes. It passes for every real model and correctly catches the deliberately leaky one.

Baselines compared: no change; last year's same-period change; continued 12-month trend; and per-series exponential smoothing (ETS).

## 4. The result: the deployed model lost to "no change"

Error relative to naive, all origins from 2018 to 2026, real DOSM data:

| Model | 1 month | 3 months | 6 months | 12 months |
|---|---:|---:|---:|---:|
| Naive: prices stay the same | 1.00 | 1.00 | 1.00 | 1.00 |
| ETS | 1.06 | 1.01 | 0.99 | 0.96 |
| **XGBoost as deployed** | **2.23** | **1.37** | **1.22** | **1.11** |

- **One month ahead,** its error was 0.68 percentage points against naive's 0.30.
- **Direction:** measured properly, it was right 56–69% of the time, not 97%.
- **Even with the leaky features,** scored exactly the way the training script did, it was 1.81× worse than naive. The leak flattered the metrics, but it wasn't the main problem.

## 5. Diagnosis

My first hypothesis was wrong. I expected tree models to *under*-predict, since they can't output values above anything seen in training and CPI keeps setting new highs. The errors actually leaned *over*: +0.3 to +0.7 points a month in 2018–2020.

Breaking errors down by category showed why. The model was worst on flat or step-shaped categories: Clothing (2.71× naive at 6 months), Alcohol & Tobacco (2.30×, which jumps when excise duty changes) and Insurance (1.76×).

**One model, predicting raw index levels, with no idea which state or category it was forecasting, learned an average upward drift and applied it everywhere.** Fixing the leak alone made it slightly *worse*, which confirmed that the problem was how the task was framed.

## 6. The redesign

- **Predict the % change, not the level.** Changes are comparable across series and over time; levels keep reaching values the model has never seen.
- **Use one model per horizon (direct forecasting).** Each model jumps straight from today to *h* months ahead. Errors don't compound, and there's no need to assume PPI and interest rates stay frozen.
- **Tell the model what it's forecasting.** State and category are categorical features.
- **Add features that match how prices actually behave:**
  - **stickiness:** about 40% of monthly category changes are exactly zero;
  - **cross-state signal:** the same category's average move across all states;
  - **seasonality:** last year's change over the same calendar months.
- **Train on absolute error,** which is what it's scored on.
- **Average it with ETS.** Combining two different methods is one of the most reliable improvements in forecasting.

| Model | 1 month | 3 months | 6 months | 12 months |
|---|---:|---:|---:|---:|
| New XGBoost model | 1.04 | 0.97 | 0.95 | 0.99 |
| **ETS + XGBoost average (production)** | 1.03 | **0.97** | **0.94** | **0.94** |
| Since 2022 (average) | 0.97 | 0.90 | 0.86 | 0.79 |

- **Direction:** the average gets the direction right 70–77% of the time.
- **Significance:** in a block bootstrap over forecast months, the new model beat naive in 91% of resamples at 3 months and 96% at 6 months.

**Being honest about the size of the gain:**
- **It's modest.** The all-period 95% intervals touch 1.0, and I evaluated two designs on the same backtest, so expect a little optimism.
- **One month ahead, nothing reliably beats naive.** Short-term inflation is mostly unpredictable beyond "about the same as now". The app now says so.

## 7. Uncertainty you can trust

The old app showed a "confidence band" of ±0.5% per month, a number with no basis. The new ranges come from the model's real backtest errors. I checked them the honest way: build the ranges from errors between August 2020 and July 2023, then see how often outcomes from August 2023 onward fell inside them.

| Approach | Coverage, target 80% (1 / 3 / 6 / 12 months) |
|---|---|
| Separate range for each category | 72% / 71% / 67% / 62% (overfit) |
| One range for all categories, 10th–90th percentile | 75% / 76% / 76% / 77% |
| **One range for all categories, 7.5th–92.5th percentile (shipped)** | **81% / 81% / 81% / 81%** |

Per-category ranges looked more informative but failed, because a category's volatility shifts over time. I shipped the version that held up.

## 8. Shipping it, and what production testing caught

**Forecasting runs as a monthly batch.** After ingest, dbt run and dbt tests, the pipeline retrains, recalibrates the ranges and publishes a forecast table. The API just looks forecasts up, re-checks for a new table hourly, and falls back to the old model if the table is missing.

**A live scorecard.** Each month, every forecast *actually published* is scored against the CPI released since. That's a growing track record that nothing could have leaked into.

Testing the real app end to end surfaced problems that unit-level checks never would have:

- **The "Reality check" panel only rendered in Bahasa Malaysia.** It was indented inside the language branch.
- **The Relocation Planner reran its work on every keystroke,** and **the app exhausted the API's rate limit** after a click or two. It sent one request per category, from a server address shared by all users. A bulk endpoint fixed it.
- **The Relocation Planner divided one state's CPI index by another's.** An index measures change since a base year, not price level, so that ratio says nothing about which state is more expensive. It now starts from survey data on household spending.
- **The "OPR" input wasn't the OPR.** The BNM dataset has no OPR series; the column was commercial banks' base rate, available only since 2022.
- **Two dependency problems:** an unpinned rebuild broke the app (pyarrow vs NumPy), and the pipeline image was missing scikit-learn. Every image is now fully pinned and built on every pull request.

## 9. What I took away

1. **Be suspicious of numbers that are too good.** A forecasting metric near 97% should prompt an investigation, not go in a README.
2. **Test the product, not the training script.** Evaluate exactly the procedure users get, at the horizons they see.
3. **Beat the simplest baseline first.** "No change" was the bar, and the original model never cleared it.
4. **Make leakage testable.** Corrupting future data and asserting nothing changes is cheap, and it settles the question.
5. **Report modest results honestly.** "3–6% better than naive, with ranges that hold 81% of the time" is a claim I can defend in any interview. "97% accurate" wasn't.

## Limitations and next steps

- **Data granularity:** state-level CPI only. DOSM's district-level cost-of-living data would make the Relocation Planner much stronger.
- **Interest rates:** a real OPR series needs a different BNM dataset.
- **Ranges:** they currently share one width across categories. A model of each category's volatility that holds up out of sample is the next improvement.
- **Track record:** the live scorecard needs months to accumulate. It's the number that ultimately matters.

*Code: `ml/training/backtest.py` (evaluation), `ml/training/forecast_batch.py` (production forecasting, calibration and scorecard). Full backtest tables: `docs/backtest_report.md`.*
