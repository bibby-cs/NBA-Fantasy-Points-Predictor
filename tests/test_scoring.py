from src.scoring import PointsLeagueScoring, fantasy_points

def test_fantasy_points_basic():
    s = PointsLeagueScoring()
    row = {"PTS": 10, "REB": 5, "AST": 4, "STL": 1, "BLK": 1, "TOV": 2, "FG3M": 3}
    fp = fantasy_points(row, s)
    assert fp != 0