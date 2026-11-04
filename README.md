# Cloud Cost Intelligence

![tests](https://github.com/Deeksha-Bapura/cloud-cost-intelligence/actions/workflows/tests.yml/badge.svg)

A cloud cost monitoring dashboard (Streamlit) and API (Flask) that flags unusual spend per service and forecasts the next month's bill. The anomaly detectors and forecasts are **measured against ground truth**, not just eyeballed.

<!-- Add after deploying: **Live demo:** https://<your-app>.streamlit.app -->

## Results

Benchmarked on 5 held-out synthetic datasets (12 months each, 6 services x 5 teams, 30 planted cost spikes per dataset). Thresholds were tuned on 5 separate validation datasets.

**Anomaly detection** — share of planted spikes caught, and false alarms raised:

| Method | Recall | Precision | False alarms / month |
|---|---|---|---|
| Static mean + 2σ (original) | 64% | 85% | 0.4 |
| Rolling z-score, weekday-aware | 83% | 95% | 0.2 |
| STL seasonal decomposition | 86% | 97% | 0.1 |
| **IsolationForest** (scikit-learn) | **88%** | **95%** | **0.2** |

**Forecast** — 30-day backtest, mean absolute percentage error (lower is better):

| Method | MAPE, last 30 days | MAPE, 3-fold average |
|---|---|---|
| 7-day moving average (original) | 13.9% | 13.3% |
| Seasonal naive (repeat last week) | 4.6% | 4.4% |
| **Holt-Winters** | **4.1%** | **3.8%** |
| Holt-Winters + spike cleaning | 4.1% | 3.5% |

In short: replacing the global 2σ rule raised spike recall from **64% to 88%** while **halving false alarms**, and Holt-Winters cut 30-day forecast error from **13.3% to 3.5%**. Full tables: [`reports/RESULTS.md`](reports/RESULTS.md).

### Why the original rule falls short

A single mean + 2σ threshold over the whole history can't tell a spike from normal variation once costs grow and follow a weekly cycle. Across the test datasets it caught:

- **28%** of small spikes (under +30% of the service's daily spend) — IsolationForest: 50%
- **44%** of weekend spikes, which sit below normal weekday cost — IsolationForest: 93%
- **44%** of spikes in the first six months, which are still below later, legitimately higher spend — IsolationForest: 89%

The improved methods compare each day with what is normal **for that service, that weekday, at that point in its growth**.

## How it works

```
analytics/
  data.py         synthetic billing data with weekly cycles, growth, a holiday dip,
                  a legitimate workload launch, noise, and planted spikes (ground truth)
  detection.py    static threshold, rolling z-score, STL, IsolationForest
  forecasting.py  moving average, seasonal naive, Holt-Winters, rolling-origin backtest
  evaluation.py   recall / precision / false alarms against the planted spikes
api/app.py        Flask API: /costs, /costs/service, /anomalies?method=, /forecast?days=
dashboard/app.py  Streamlit dashboard
scripts/
  generate_data.py   writes data/cloud_usage.csv and data/planted_anomalies.csv
  evaluate.py        runs the benchmark, writes reports/results.json and RESULTS.md
tests/            pytest suite (data, detectors, forecasts, API, dashboard smoke test)
```

**Detectors** (each runs per service, per day; only upward spikes are flagged):

- **Rolling z-score, weekday-aware** — compares today's cost with the same weekday over the last 4 weeks, scaled by the recent robust spread. Uses only past data, so it works as a live alert.
- **STL** — splits each service's cost into trend, weekly season and residual, and flags large residuals. Uses the full history (batch review).
- **IsolationForest** — scikit-learn model over engineered features (ratio to same weekday, ratio to last week's median, day-over-day change, STL residual), standardised per service so one model covers small and large services. Its `contamination` setting is the alert budget.

**Evaluation** — a spike counts as caught if any of its days is flagged; every alert on a normal service-day is a false alarm. Each method's threshold is chosen on validation datasets (seeds 0-4) and scored on different test datasets (seeds 42-46), so nothing is tuned on its own test data. The static rule gets the same tuning; 2σ is already its best setting.

**Forecast backtest** — hide the last 30 days, forecast them from the history before, measure MAPE; repeat for the two 30-day windows before that.

## Run it

```bash
pip install -r requirements-dev.txt

python scripts/generate_data.py      # regenerate the demo data
python scripts/evaluate.py           # re-run the benchmark (~40 s)
pytest                               # run the tests
streamlit run dashboard/app.py       # open the dashboard
python api/app.py                    # API on http://localhost:10000
```

The dashboard reads `data/cloud_usage.csv` directly. To read from the API instead, set `COST_API_URL` (for example `https://cloud-cost-intelligence.onrender.com`).

## Deploy the dashboard (free)

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io) with GitHub.
2. **Create app** → pick this repo, branch `main`, main file `dashboard/app.py`.
3. Deploy, then paste the `*.streamlit.app` link at the top of this README.

## Data

All data is **synthetic** (`scripts/generate_data.py`), generated to look like a year of AWS spend: 6 services (EC2, EKS, RDS, S3, Lambda, CloudFront), 5 teams, weekday/weekend cycles, 2-6% monthly growth per service, a December holiday dip, a backend workload moving to EKS mid-year (a legitimate step change, not an anomaly), and 30 cost spikes of +10% to +100% of a service's daily spend lasting 1-3 days. The planted spikes are saved to `data/planted_anomalies.csv`.
