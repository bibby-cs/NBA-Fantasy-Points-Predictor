import os
import sys
import joblib
import pandas as pd
import streamlit as st

# Ensure imports from src work when running via `streamlit run app/streamlit_app.py`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

@st.cache_resource
def load_models():
    mid = joblib.load("models/model_mid.joblib")
    low = joblib.load("models/model_low.joblib")
    high = joblib.load("models/model_high.joblib")
    cols = joblib.load("models/feature_columns.joblib")
    return mid, low, high, cols

@st.cache_data
def load_data():
    # Use processed features by default
    return pd.read_parquet("data/processed/features_2024-25.parquet")

st.title("NBA Fantasy Points Predictor")

df = load_data()
mid, low, high, cols = load_models()

# Player selector using a searchable selectbox
# Player selector using a searchable selectbox
df["playerName"] = df["firstName"] + " " + df["familyName"]
player_map = df[["personId", "playerName"]].drop_duplicates().sort_values("playerName")

player_name = st.selectbox("Player", player_map["playerName"].tolist())
player_id = int(player_map[player_map["playerName"] == player_name]["personId"].iloc[0])

# Filter data for selected player
player_games = df[df["personId"] == player_id].sort_values("GAME_DATE")
# Filter data for selected player
player_games = df[df["personId"] == player_id].sort_values("GAME_DATE")

# Latest row and features
latest = player_games.iloc[-1]
x = player_games[cols].fillna(0)
x_latest = x.iloc[[-1]]

# Predictions
pred_mid = float(mid.predict(x_latest)[0])
pred_low = float(low.predict(x_latest)[0])
pred_high = float(high.predict(x_latest)[0])

# Display
c1, c2, c3 = st.columns(3)
c1.metric("Next Game Prediction", f"{pred_mid:.1f}")
c2.metric("Low (10th pct)", f"{pred_low:.1f}")
c3.metric("High (90th pct)", f"{pred_high:.1f}")

st.subheader("Recent fantasy points")
st.line_chart(player_games.set_index("GAME_DATE")["FANTASY_PTS"])

st.subheader("Latest feature values")
st.dataframe(x_latest.T.rename(columns={x_latest.index[0]: "value"}).head(40))