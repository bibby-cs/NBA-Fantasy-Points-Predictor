from pathlib import Path
import joblib
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from src.features import add_rolling_features

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def train(df: pd.DataFrame):
    df = add_rolling_features(df)

    target = "TARGET_NEXT_FANTASY_PTS"
    drop_cols = [
        target,
        "FANTASY_PTS",
        "GAME_ID",
        "GAME_DATE",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "TEAM_CITY",
        "PLAYER_NAME",
        "COMMENT",
        "START_POSITION",
        "NICKNAME",
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target]

    # Keep only numeric features
    X = X.select_dtypes(include=["number"]).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mid = GradientBoostingRegressor(loss="squared_error", random_state=42)
    low = GradientBoostingRegressor(loss="quantile", alpha=0.10, random_state=42)
    high = GradientBoostingRegressor(loss="quantile", alpha=0.90, random_state=42)

    mid.fit(X_train, y_train)
    low.fit(X_train, y_train)
    high.fit(X_train, y_train)

    pred = mid.predict(X_test)
    mae = mean_absolute_error(y_test, pred)

    joblib.dump(mid, MODEL_DIR / "model_mid.joblib")
    joblib.dump(low, MODEL_DIR / "model_low.joblib")
    joblib.dump(high, MODEL_DIR / "model_high.joblib")
    joblib.dump(list(X.columns), MODEL_DIR / "feature_columns.joblib")

    return {"mae": mae, "n_train": len(X_train), "n_test": len(X_test)}

if __name__ == "__main__":
    df = pd.read_parquet("data/raw/boxscores_2024-25_Regular_Season.parquet")
    metrics = train(df)
    print(metrics)