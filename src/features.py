import pandas as pd

def add_rolling_features(df: pd.DataFrame, windows=(3, 5, 10)) -> pd.DataFrame:
    df = df.copy()
    df["FANTASY_PTS"] = (
        df["points"]
        + 1.2 * df["reboundsTotal"]
        + 1.5 * df["assists"]
        + 3 * df["steals"]
        + 3 * df["blocks"]
        - df["turnovers"]
    )
    
    

    
    # Ensure ordering
    df["GAME_DATE"] = pd.to_datetime(df["gameId"].astype(str).str[-5:], errors="coerce")
    df = df.sort_values(["personId", "GAME_DATE", "gameId"])

    base_cols = [
        "FANTASY_PTS", "minutes", "points", "reboundsTotal", "assists", "steals", "blocks", "turnovers", "threePointersMade"
    ]
    for c in ["minutes", "points", "reboundsTotal", "assists", "steals", "blocks", "turnovers", "threePointersMade", "FANTASY_PTS"]:
        df[c] = df[c].astype(str).str.replace(":", ".").str.extract(r"([\d\.]+)")[0]
        df[c] = pd.to_numeric(df[c], errors="coerce")


    for w in windows:
        for c in base_cols:
            df[f"{c}_roll{w}_mean"] = (
                df.groupby("personId")[c]
                  .shift(1)
                  .rolling(w, min_periods=1)
                  .mean()
                  .reset_index(level=0, drop=True)
            )
            df[f"{c}_roll{w}_std"] = (
                df.groupby("personId")[c]
                  .shift(1)
                  .rolling(w, min_periods=1)
                  .std()
                  .reset_index(level=0, drop=True)
            )

    # Target: next game fantasy points
    df["TARGET_NEXT_FANTASY_PTS"] = df.groupby("personId")["FANTASY_PTS"].shift(-1)

    # Drop rows where target is missing
    df = df.dropna(subset=["TARGET_NEXT_FANTASY_PTS"])

    return df

if __name__ == "__main__":
    df = pd.read_parquet("data/raw/boxscores_2024-25_Regular_Season.parquet")
    df = add_rolling_features(df)
    df.to_parquet("data/processed/features_2024-25.parquet", index=False)
    print("Saved processed feature file successfully.")