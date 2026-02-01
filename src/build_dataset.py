from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.data_fetch import fetch_game_ids, fetch_boxscore_players
from src.scoring import PointsLeagueScoring, fantasy_points

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def build_season(season: str, season_type: str = "Regular Season", max_games: int | None = None) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    game_ids = fetch_game_ids(season=season, season_type=season_type)
    if max_games:
        game_ids = game_ids[:max_games]

    rows = []
    for gid in tqdm(game_ids, desc=f"Fetching {season} {season_type}"):
        try:
            dfp = fetch_boxscore_players(gid)
            rows.append(dfp)
        except Exception:
            continue

    df = pd.concat(rows, ignore_index=True)

    s = PointsLeagueScoring()
    df["FANTASY_PTS"] = df.apply(lambda r: fantasy_points(r, s), axis=1)

    out_raw = RAW_DIR / f"boxscores_{season}_{season_type.replace(' ', '_')}.parquet"
    df.to_parquet(out_raw, index=False)

    return df

if __name__ == "__main__":
    df = build_season("2024-25", max_games=200)
    print(df.shape)