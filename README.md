# 🏀 NBA Fantasy Points Predictor

A complete end-to-end machine learning pipeline that predicts NBA player fantasy points using live data from the [NBA API](https://github.com/swar/nba_api).  
Built with **Python**, **pandas**, **scikit-learn**, and **Streamlit**.

---

## 📂 Project Structure

NBA-FANTASY-POINTS-PREDICTOR/ │ ├── app/ │ └── streamlit_app.py # Streamlit dashboard │ ├── data/ │ ├── raw/ # Raw data from NBA API │ └── processed/ # Processed feature data │ ├── models/ # Trained models │ ├── src/ │ ├── init.py │ ├── data_fetch.py # Fetches data from NBA API │ ├── features.py # Builds rolling and fantasy features │ ├── train.py # Trains predictive models │ ├── backtest.py # Evaluates model performance │ ├── explain.py # Feature importance analysis │ ├── scoring.py # Scoring utilities │ ├── utils.py # Helper functions │ └── config.py # Configuration constants │ ├── tests/ │ └── test_scoring.py # Unit tests │ ├── requirements.txt # Dependencies ├── README.md # Project documentation └── .gitignore


---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/NBA-Fantasy-Points-Predictor.git
cd NBA-Fantasy-Points-Predictor
2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
3. Install dependencies
pip install -r requirements.txt
🧠 Pipeline Overview
Step 1: Fetch Data
Fetches all NBA boxscore data for the 2024–25 season using the NBA API.

python src/data_fetch.py
Output:

data/raw/boxscores_2024-25_Regular_Season.parquet
Step 2: Build Features
Generates fantasy points and rolling averages for each player.

python src/features.py
Output:

data/processed/features_2024-25.parquet
Step 3: Train Model
Trains a Random Forest model to predict fantasy points.

python src/train.py
Output:

models/fantasy_model.pkl
Step 4: Backtest
Evaluates model performance on unseen data.

python src/backtest.py
Step 5: Explain Model
Shows feature importances using permutation or SHAP values.

python src/explain.py
Step 6: Run Streamlit App
Launches the interactive dashboard.

streamlit run app/streamlit_app.py
🖥️ Streamlit App Features
Search and select any NBA player
View predicted fantasy points for the next game
See recent fantasy point trends
Inspect latest feature values used for prediction
📊 Fantasy Points Formula
Fantasy points are computed as:

Fantasy Points = PTS + 1.2×REB + 1.5×AST + 3×STL + 3×BLK − TOV

🧪 Testing
Run all tests:

pytest
🛠️ Technologies Used
Python 3.9+
pandas, numpy
scikit-learn
streamlit
nba_api
joblib
matplotlib
🚀 Future Improvements
Add live game updates
Integrate player injury and lineup data
Deploy app on Streamlit Cloud or AWS
Add model retraining automation
👤 Author
Bisayo Awude
GitHub: @bibby-cs

📜 License
This project is licensed under the MIT License.

