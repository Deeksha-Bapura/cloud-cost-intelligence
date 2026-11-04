import pytest

from api.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json["rows"] > 0


def test_cost_by_service(client):
    r = client.get("/costs/service")
    assert r.status_code == 200
    assert {"service", "cost"} <= set(r.json[0])


@pytest.mark.parametrize("method", ["isolation_forest", "rolling_zscore", "stl", "static"])
def test_anomalies(client, method):
    r = client.get(f"/anomalies?method={method}")
    assert r.status_code == 200
    assert r.json["method"] == method
    assert all({"date", "service", "cost", "score"} <= set(a) for a in r.json["anomalies"])


def test_anomalies_unknown_method(client):
    assert client.get("/anomalies?method=magic").status_code == 400


def test_forecast(client):
    r = client.get("/forecast?days=14")
    assert r.status_code == 200
    assert len(r.json["forecast"]) == 14
