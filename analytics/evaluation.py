"""Score anomaly detectors against planted ground truth."""
from __future__ import annotations

import pandas as pd


def score_detector(flags: pd.DataFrame, labels: pd.DataFrame, days: int) -> dict:
    """Compare detector output with planted anomalies.

    flags:  date, service, flag   (one row per service-day)
    labels: event_id, date, service (one row per planted anomaly-day)

    recall          share of planted events caught (an event counts if any of its days is flagged)
    precision       share of alerts that point at a real anomaly-day
    false_alarms    alerts on normal service-days
    """
    truth = set(zip(labels["date"], labels["service"]))
    alerts = flags.loc[flags["flag"], ["date", "service"]]
    alert_keys = set(zip(alerts["date"], alerts["service"]))

    hits = alert_keys & truth
    false_alarms = len(alert_keys - truth)

    lab = labels.assign(hit=[(d, s) in alert_keys for d, s in zip(labels["date"], labels["service"])])
    events = lab.groupby("event_id")["hit"].any()

    recall = float(events.mean()) if len(events) else 0.0
    precision = len(hits) / len(alert_keys) if alert_keys else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "events_caught": int(events.sum()),
        "events_total": int(len(events)),
        "alerts": len(alert_keys),
        "false_alarms": false_alarms,
        "false_alarms_per_month": false_alarms / (days / 30.4),
    }
