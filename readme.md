# Football Corners Forecast

A machine learning project that predicts total corners in football matches using historical data from 8 European leagues.

Predicts **total corners** in matches with **MAE < 2.0** (average error less than 2 corners) to support betting analysis.

---

## 📊 Dataset

### Data Source
- **Platform**: FBref.com (via `soccerdata` library)
- **Leagues**: Premier League 🏴󠁧󠁢󠁥󠁮󠁧󠁿, La Liga 🇪🇸, Bundesliga 🇩🇪, Ligue 1 🇫🇷, Serie A 🇮🇹, Eredivisie 🇳🇱, Primeira Liga 🇵🇹, Pro League 🇧🇪
- **Seasons**: 2017-2025
- **Total Matches**: ~21,000

### Features Collected
- **Shooting**: xG, shots, shots on target, distance
- **Passing**: corners, passes, long passes, assists
- **Defense**: tackles, blocks, interceptions, clearances
- **Possession**: touches, carries, possession %
- **Goalkeeping**: save %

---

## Data Processing

### 1. Data Collection
```
FBref.com → Download stats → Merge leagues → Clean data → CSV
```

![alt text](https://github.com/danielsaed/futbol_corners_forecast/blob/main/img/example_fbstats.jpg?raw=true)


**Output**: dataset_cleaned.csv

### 2. Feature Engineering

Created **80+ features** per match:

| Category | Features | Examples |
|----------|----------|----------|
| **Team Averages** | 32 | Home/away avg corners, xG, shots |
| **Opponent Stats** | 16 | Performance vs each team |
| **Head-to-Head** | 3 | Last 3 matches between teams |
| **Form & Variance** | 8 | Recent form, consistency |
| **League Encoding** | 8 | One-hot encoded leagues |
| **Advanced Metrics** | 15 | Shot accuracy, offensive intensity |

**Key engineered features**:
```python
- sh_accuracy = shots_on_target / total_shots
- offensive_index = (goals + xG) × shot_accuracy
- attacking_presence = touches_att_3rd / total_touches
- high_press_intensity = tackles_att_3rd / total_tackles
```

**Output**: dataset_processed.csv

---

## Model

### Algorithm: **XGBoost Regressor**

**Why XGBoost?**
- Handles non-linear relationships
- Works well with 80+ features
- Resistant to overfitting
- Fast training/prediction

### Training Process

```
Total: 21,000 matches
├── Train (70%):      14,700 matches
├── Validation (15%):  3,150 matches
└── Test (15%):        3,150 matches
```

**Hyperparameters** (found via GridSearchCV):
```python
{
    'n_estimators': 200,
    'max_depth': 4,
    'learning_rate': 0.02,
    'reg_alpha': 5.0,
    'reg_lambda': 8.0,
    'subsample': 0.7,
    'colsample_bytree': 0.7
}
```

---

## 📈 Results

### Model Performance

| Set | MAE | R² | RMSE |
|-----|-----|-----|------|
| **Train** | 1.65 | 0.52 | 2.21 |
| **Validation** | 1.82 | 0.48 | 2.35 |
| **Test** | **1.85** | **0.46** | **2.38** |

✅ **Test MAE = 1.85**: Predictions are off by **1.85 corners** on average

### Error Distribution

```
Errors < 1 corner:    42%
Errors < 1.5 corners: 58%
Errors < 2 corners:   74%
Errors < 3 corners:   91%
```

### Top 10 Most Important Features

| Feature | Importance | Description |
|---------|------------|-------------|
| `lst_team1_home_avg_ck` | 0.0842 | Home team avg corners at home |
| `lst_team2_away_avg_ck` | 0.0795 | Away team avg corners away |
| `lst_team1_home_xg` | 0.0623 | Home team expected goals |
| `lst_h2h_avg_ck` | 0.0581 | Head-to-head avg corners |
| `lst_team1_home_sh` | 0.0534 | Home team shots |
| `lst_team2_away_xg` | 0.0489 | Away team expected goals |

---

## Prediction System

### Input
```python
predict_corners(
    local="Barcelona",
    visitante="Real Madrid",
    jornada=15,
    temporada="2526",
    league_code="ESP"
)
```

### Output Example
```
🏟️  Barcelona vs Real Madrid
📅 Season 2526 | Round 15

🎯 PREDICTION: 10.3 corners
📊 Most probable: 10 corners (12.5%)
📊 80% confidence: 7-13 corners

🎯 OVER/UNDER PROBABILITIES:
Over 8.5:  72.3% @1.38 - HIGH ✅
Over 9.5:  58.1% @1.72 - MEDIUM ⚠️
Over 10.5: 43.2% @2.31 - LOW ❌

⚠️ RELIABILITY: VERY HIGH ⭐⭐⭐ (Score: 71/100)
```

### Reliability Score

Measures team consistency:
```
Score = (100 - CV) × 0.4 + 
        consistency × 0.3 + 
        trend_stability × 0.3

- Score > 65: VERY HIGH ⭐⭐⭐
- Score > 50: HIGH ⭐⭐
- Score > 35: MEDIUM ⭐
- Score < 35: LOW ⚠️
```

---

## Project Structure

```
futbol_corners_forecast/
│
├── config/
│   └── model_config.json          # Best hyperparameters
│
├── dataset/
│   ├── cleaned/
│   │   └── dataset_cleaned.csv    # Raw processed data
│   └── processed/
│       └── dataset_processed.csv  # ML-ready features
│
├── models/
│   ├── xgboost_corners_*.pkl      # Trained model
│   ├── scaler_corners_*.pkl       # Feature scaler
│   └── feature_importance_*.csv   # Feature rankings
│
├── mlruns/                        # MLflow experiments
│
├── src/
│   ├── models/
│   │   ├── train_model.py         # Training pipeline
│   │   └── test_model.py          # Prediction system
│   │
│   └── process_data/
│       ├── generate_dataset.py    # Data collection
│       └── process_dataset.py     # Feature engineering
│
├── EDA.ipynb                      # Exploratory analysis
└── README.md
```

---

## Technologies

- **Data**: `pandas`, `numpy`, `soccerdata`
- **ML**: `XGBoost`, `scikit-learn`
- **Tracking**: `MLflow`
- **Statistics**: `scipy` (Poisson distribution)
- **Visualization**: `matplotlib`, `plotly`

---

## Key Findings

### What Works Well ✅
- Consistent teams → Better predictions (MAE ~1.6)
- Top leagues → More data = Better accuracy
- Mid-season matches → More historical data

### Challenges ⚠️
- Inconsistent teams → Higher error (MAE ~2.3)
- Early season → Limited historical data
- Defensive matches → Fewer corners = harder to predict



---

## License

Educational purposes only. Not financial advice.

