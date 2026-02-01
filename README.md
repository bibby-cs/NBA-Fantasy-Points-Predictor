NBA Fantasy Points Predictor

What it does
- Predicts a player's next game fantasy points from public NBA box score data
- Outputs a prediction range (10th to 90th percentile)
- Provides feature importance
- Includes backtesting and a Streamlit demo UI

Stack
- Python, pandas, scikit-learn, Streamlit

Quickstart
1) Install
pip install -r requirements.txt

2) Build data
python -m src.build_dataset

3) Train models
python -m src.train

4) Run app
streamlit run app/streamlit_app.py

Notes
- This project uses rolling, past-only features.
- Models are saved in /models.