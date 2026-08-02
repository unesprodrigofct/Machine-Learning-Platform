import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_platform.plugins.evaluators.metrics import evaluate


def test_classification_evaluation_returns_expected_metrics() -> None:
    features = pd.DataFrame({"feature": [1, 2, 3, 4]})
    target = pd.Series([0, 0, 1, 1])
    model = DummyClassifier(strategy="most_frequent").fit(features, target)

    metrics = evaluate(model, features, target, "classification", ["accuracy", "f1_weighted"])

    assert set(metrics) == {"accuracy", "f1_weighted"}


def test_regression_evaluation_returns_expected_metrics() -> None:
    features = pd.DataFrame({"feature": [1, 2, 3, 4]})
    target = pd.Series([1.0, 2.0, 3.0, 4.0])
    model = DummyRegressor(strategy="mean").fit(features, target)

    metrics = evaluate(model, features, target, "regression", ["mae", "rmse", "r2"])

    assert set(metrics) == {"mae", "rmse", "r2"}


def test_anomaly_evaluation_returns_anomaly_rate() -> None:
    features = pd.DataFrame({"feature": [1, 2, 3, 4, 50]})
    model = IsolationForest(random_state=42).fit(features)

    metrics = evaluate(model, features, None, "anomaly_detection", ["anomaly_rate"])

    assert 0 <= metrics["anomaly_rate"] <= 1


def test_clustering_evaluation_returns_core_metrics() -> None:
    features = pd.DataFrame({"x": [1, 2, 10, 11], "y": [1, 2, 10, 11]})
    model = Pipeline([("preprocessor", StandardScaler()), ("model", KMeans(n_clusters=2, random_state=42, n_init=10))]).fit(features)

    metrics = evaluate(
        model,
        features,
        None,
        "clustering",
        ["silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"],
    )

    assert {"silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"} == set(metrics)
