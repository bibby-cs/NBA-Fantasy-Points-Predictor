import time
from typing import List
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog, boxscoretraditionalv3

def fetch_game_ids(season: str, season_type: str = "Regular Season") -> List[str]:
    season_type = "Regular Season"
    lg = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=season_type
    )
    df = lg.get_data_frames()[0]
    return sorted(df["GAME_ID"].unique().tolist())

def fetch_boxscore_players(game_id: str, sleep_s: float = 2.0, retries: int = 5) -> pd.DataFrame:
    import requests
    for attempt in range(retries):
        try:
            time.sleep(sleep_s)
            bs = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id, timeout=60)
            players = bs.player_stats.get_data_frame()
            players["GAME_ID"] = game_id
            return players
        except requests.exceptions.ReadTimeout:
            print(f"Timeout on {game_id}, retrying ({attempt+1}/{retries})...")
    print(f"Failed to fetch {game_id} after {retries} retries.")
    return pd.DataFrame()

if __name__ == "__main__":
    season = "2024-25"
    game_ids = fetch_game_ids(season)
    print(f"Fetched {len(game_ids)} game IDs")
    all_players = []
    for gid in game_ids:
        try:
            players = fetch_boxscore_players(gid, sleep_s=3.5)
            if not players.empty:
                all_players.append(players)
            else:
                print(f"No data for {gid}")
            time.sleep(4)
        except Exception as e:
            print(f"Error fetching {gid}: {e}")
            time.sleep(5)
    df = pd.concat(all_players, ignore_index=True)
    df.to_parquet("data/raw/boxscores_2024-25_Regular_Season.parquet", index=False)
    print("Saved parquet file successfully.")