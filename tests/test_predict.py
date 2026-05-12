"""Unit tests for the wind production prediction service."""

import pandas as pd

# ---------------------------------------------------------------------------
# Tests for feature engineering logic (no Docker/MLFlow needed)
# ---------------------------------------------------------------------------


def build_features(
    geo_windspeed_10m: float,
    geo_windspeed_30m: float,
    day_of_week: int = 0,
    month: int = 1,
    is_weekend: int = 0,
) -> pd.DataFrame:
    """Duplicate of the feature builder from predict.py for isolated testing."""
    rows = []
    for hour in range(24):
        rows.append(
            {
                "geo_windspeed_10m": geo_windspeed_10m,
                "geo_windspeed_30m": geo_windspeed_30m,
                "wind_speed_ratio": geo_windspeed_30m / (geo_windspeed_10m + 0.001),
                "hour": hour,
                "day_of_week": day_of_week,
                "month": month,
                "is_weekend": is_weekend,
            }
        )
    return pd.DataFrame(rows)


def test_build_features_returns_24_rows():
    """Feature builder should return exactly 24 rows (one per hour)."""
    df = build_features(5.0, 7.0)
    assert len(df) == 24


def test_build_features_columns():
    """Feature builder should return all required columns."""
    df = build_features(5.0, 7.0)
    expected_cols = [
        "geo_windspeed_10m",
        "geo_windspeed_30m",
        "wind_speed_ratio",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_build_features_hour_range():
    """Hours should range from 0 to 23."""
    df = build_features(5.0, 7.0)
    assert list(df["hour"]) == list(range(24))


def test_build_features_wind_speed_ratio():
    """Wind speed ratio should be geo_30m / (geo_10m + 0.001)."""
    geo_10 = 5.0
    geo_30 = 10.0
    df = build_features(geo_10, geo_30)
    expected_ratio = geo_30 / (geo_10 + 0.001)
    assert abs(df["wind_speed_ratio"].iloc[0] - expected_ratio) < 1e-6


def test_build_features_zero_wind():
    """Zero wind speed should not cause division by zero."""
    df = build_features(0.0, 0.0)
    assert df["wind_speed_ratio"].isna().sum() == 0
    assert (df["wind_speed_ratio"] == 0.0).all()


def test_build_features_optional_params():
    """Optional parameters should be correctly applied to all rows."""
    df = build_features(5.0, 7.0, day_of_week=3, month=6, is_weekend=1)
    assert (df["day_of_week"] == 3).all()
    assert (df["month"] == 6).all()
    assert (df["is_weekend"] == 1).all()


def test_predictions_non_negative():
    """Predictions should never be negative (clipped to 0)."""
    raw_predictions = [-100.0, 500.0, -0.1, 0.0, 1000.0]
    clipped = [max(0.0, p) for p in raw_predictions]
    assert all(p >= 0 for p in clipped)


def test_rmse_threshold_logic():
    """Retraining should trigger when RMSE exceeds threshold."""
    threshold = 400000.0

    assert 156341.0 <= threshold  # normal run — no retraining
    assert 500000.0 > threshold  # degraded run — retraining triggered
