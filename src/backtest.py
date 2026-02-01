import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from src.features import add_rolling_features

def backtest(df: pd.DataFrame, min_train_date: str, split_date: str):
    df = df.copy()
    if "GAME_DATE" not in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["gameId"].astype(str).str[-5:], errors="coerce")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    df = add_rolling_features(df)
    target = "TARGET_NEXT_FANTASY_PTS"

    X_all = df.select_dtypes(include=["number"]).fillna(0)
    y_all = df[target]

    split_date = df["GAME_DATE"].quantile(0.8)
    train_mask = df["GAME_DATE"] < split_date
    test_mask = df["GAME_DATE"] >= split_date

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    mid = GradientBoostingRegressor(loss="squared_error", random_state=42)
    low = GradientBoostingRegressor(loss="quantile", alpha=0.10, random_state=42)
    high = GradientBoostingRegressor(loss="quantile", alpha=0.90, random_state=42)

    mid.fit(X_train, y_train)
    low.fit(X_train, y_train)
    high.fit(X_train, y_train)

    pred_mid = mid.predict(X_test)
    pred_low = low.predict(X_test)
    pred_high = high.predict(X_test)

    mae = mean_absolute_error(y_test, pred_mid)
    coverage = ((y_test >= pred_low) & (y_test <= pred_high)).mean()

    return {
        "mae": float(mae),
        "coverage_80pct": float(coverage),
        "n_test": int(test_mask.sum()),
    }

if __name__ == "__main__":
    df = pd.read_parquet("data/raw/boxscores_2024-25_Regular_Season.parquet")
    print(backtest(df, min_train_date="2024-10-01", split_date="2025-01-01"))