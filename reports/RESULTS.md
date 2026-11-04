# Benchmark results

Each dataset is 365 days of synthetic cost data for 6 services x 5 teams, with weekly cycles, growth, a holiday dip, a legitimate workload launch, and 30 planted cost spikes (+10% to +100% of the service's daily spend, lasting 1-3 days).

Thresholds were tuned on validation datasets (seeds [0, 1, 2, 3, 4]) and scored on separate test datasets (seeds [42, 43, 44, 45, 46]). Numbers are averages over the test datasets.

## Anomaly detection (per service, per day)

| Method | Setting | Recall (spikes caught) | Precision | F1 | False alarms / month |
|---|---|---|---|---|---|
| Static mean + kσ (original) | k=2.0 | 64% (96/150) | 85% | 0.73 | 0.4 |
| Rolling z-score (weekday-aware) | k=4.0 | 83% (125/150) | 95% | 0.88 | 0.2 |
| STL seasonal decomposition | k=8.0 | 86% (129/150) | 97% | 0.91 | 0.1 |
| IsolationForest | contamination=0.025 | 88% (132/150) | 95% | 0.91 | 0.2 |

## Forecast backtest (total daily cost, 30-day horizon)

MAPE = mean absolute percentage error. 'Last 30 days' hides the final month and forecasts it; '3-fold' averages three consecutive 30-day holdouts (rolling origin).

| Method | MAPE, last 30 days | MAPE, 3-fold average |
|---|---|---|
| 7-day moving average (original) | 13.9% | 13.3% |
| Seasonal naive (last week) | 4.6% | 4.4% |
| Holt-Winters | 4.1% | 3.8% |
| Holt-Winters + spike cleaning | 4.1% | 3.5% |
