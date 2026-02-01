import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

def permutation_importance_report(model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 10):
    r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42)
    imp = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std,
    }).sort_values("importance_mean", ascending=False)
    return imp

if __name__ == "__main__":
    model = joblib.load("models/model_mid.joblib")
    cols = joblib.load("models/feature_columns.joblib")

    df = pd.read_parquet("data/raw/boxscores_2024-25_Regular_Season.parquet")
    from src.features import add_rolling_features
    df = add_rolling_features(df)

    target = "TARGET_NEXT_FANTASY_PTS"
    X = df[cols].fillna(0)
    y = df[target]

    imp = permutation_importance_report(model, X, y)
    print(imp.head(20))