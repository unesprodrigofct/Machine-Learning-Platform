import pandas as pd

from ml_platform.domain.config import TrainingConfig
from ml_platform.plugins.validation.strategies import create_split


def test_last_n_months_keeps_the_latest_period_out_of_training() -> None:
    config = TrainingConfig.model_validate({
        "name": "temporal-split",
        "problem_type": "unsupervised",
        "task": "clustering",
        "data_source": {"type": "csv", "path": "unused.csv"},
        "validation": {
            "strategy": "out_of_time_last_n_months",
            "date_column": "event_date",
            "validation_months": 3,
        },
        "model": {"algorithm": "kmeans"},
    })
    dataframe = pd.DataFrame({
        "event_date": ["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01", "2025-05-01"],
        "feature": [1, 2, 3, 4, 5],
    })

    split = create_split(dataframe, config)

    assert split.train["event_date"].max() == "2025-01-01"
    assert split.validation["event_date"].min() == "2025-02-01"


def test_out_of_time_range_uses_only_historical_data_for_training() -> None:
    config = TrainingConfig.model_validate({
        "name": "range-split",
        "problem_type": "unsupervised",
        "task": "clustering",
        "data_source": {"type": "csv", "path": "unused.csv"},
        "validation": {
            "strategy": "out_of_time_range",
            "date_column": "event_date",
            "validation_start_date": "2025-03-01",
            "validation_end_date": "2025-03-31",
        },
        "model": {"algorithm": "kmeans"},
    })
    dataframe = pd.DataFrame({"event_date": ["2025-01-01", "2025-02-01", "2025-03-01"], "feature": [1, 2, 3]})

    split = create_split(dataframe, config)

    assert split.train["event_date"].tolist() == ["2025-01-01", "2025-02-01"]
    assert split.validation["event_date"].tolist() == ["2025-03-01"]
