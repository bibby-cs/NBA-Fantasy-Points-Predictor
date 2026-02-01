from dataclasses import dataclass

@dataclass(frozen=True)
class PointsLeagueScoring:
    # Default: simple points, change later to match your league
    pts: float = 1.0
    reb: float = 1.2
    ast: float = 1.5
    stl: float = 3.0
    blk: float = 3.0
    tov: float = -1.0
    fg3m: float = 0.5

def fantasy_points(row, s: PointsLeagueScoring) -> float:
    return (
        row.get("PTS", 0) * s.pts
        + row.get("REB", 0) * s.reb
        + row.get("AST", 0) * s.ast
        + row.get("STL", 0) * s.stl
        + row.get("BLK", 0) * s.blk
        + row.get("TOV", 0) * s.tov
        + row.get("FG3M", 0) * s.fg3m
    )