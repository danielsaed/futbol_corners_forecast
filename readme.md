# Football Corners Forecast

[![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-WEB-181717?style=flat)](https://daniel-saed-corner-forecast.hf.space/)
[![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-API-FFD21E?style=flat)](https://huggingface.co/spaces/daniel-saed/futbol-corners-forecast-api)
    

A machine learning project that predicts total corners in football matches using historical data from 8 European leagues.

Predicts **total corners** in matches with **MAE < 2.0** (average error less than 2 corners) to support betting analysis.

- **API:** https://huggingface.co/spaces/daniel-saed/futbol-corners-forecast-api

- **Web:** https://daniel-saed-corner-forecast.hf.space/
--- 


## Web Page

![alt text](https://github.com/danielsaed/futbol_corners_forecast/blob/main/img/web_page.jpg?raw=true)

## Technologies

- **Data**: `pandas`, `numpy`, `soccerdata`
- **ML**: `XGBoost`, `scikit-learn`
- **Tracking**: `MLflow`
- **Statistics**: `scipy` (Poisson distribution)
- **Visualization**: `matplotlib`, `plotly`

---

## 📊 Dataset

### Data Source
- **Platform**: FBref.com (via `soccerdata` library)
- **Leagues**: Premier League, La Liga, Bundesliga, Ligue 1, Serie A, Eredivisie, Primeira Liga, Pro League
- **Seasons**: 2018-2025 (years where advanced data is available)
- **Total Matches**: ~21,000

### Features Collected
- **Shooting**: xG, shots, shots on target, shot distance, shot creation actions
- **Passing**: corners, crosses, passes (total attemps, progressive, last 1/3, long passes), assists
- **Defense**: tackles (Total, last 1/3), blocks, interceptions, clearances
- **Possession**: touches, carries (progressive, last 1/3, penalty area), possession %
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

**Basic features used (9)**:

Processed with averages of their own leagues, example Average corners  = Average corners team - Average corners league
```python
- Average corners
- Varianze corners
- Average Xg
- Average sca
- Average crosses
- Average possession
- Average attemps in 1/3
- Average GF
- Average GA
```

**Advanced Key engineered features used (15)**:
```python
SHOTS

- shot accuracy
- xg shot
- possession_shot

PASSES

- progressive_pass_ratio
- final_third_involvement
- assist_sca
- creative_efficiency

DEFENSE

- interception_tackle
- clearance_ratio
- high press intensity

POSSESSION

- progressive_carry_ratio
- carry_pass_balance
- transition_index

ATTACK

- offensive index
- attacking presence

```

**Other features (11)**:
```python
POINTS PER GAME

- average points per game local team
- average points per game visit team
- difference poinst per game


LEAGUES ONE HOT ENCODING

- premier league
- ligue 1
- bundesliga
- la liga
- eredivise
- serie a
- primeira liga, 
- pro league

```

<br>

### Created **269 features** per match:

<br>

| Category | Features | Examples |
|----------|----------|----------|
| **Local Team Averages** | 96 | Form, General (Home/away) - Basic  + Advance features|
| **Visit Team Averages** | 96 | Form, General (Home/away) - Basic  + Advance features  |
| **Head-to-Head Averages** | 48 | Last 3 matches (Home/away) - Basic  + Advance features |
| **Points Per Game Features** | 3 | Poinst Local, Visit and Difference |
| **League Encoding** | 8 | One-hot encoded leagues |
| **Team against Averages** | 18 | Basic features against teams |



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

MLFlow image

![alt text](https://github.com/danielsaed/futbol_corners_forecast/blob/main/img/Parameters.jpg?raw=true)

```python
{
    'n_estimators': 200,
    'max_depth': 4,
    'learning_rate': 0.03,
    'reg_alpha': 3.0,
    'reg_lambda': 5.0,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'colsample_bylevel': 0.6,
    'best_gamma':1.0
}
```

---

## 📈 Results

### Model Performance
MLFlow image

![alt text](https://github.com/danielsaed/futbol_corners_forecast/blob/main/img/Metrics.jpg?raw=true)



| Set | MAE | R² | RMSE |
|-----|-----|-----|------|
| **Train** | 1.78 | 0.49 | 2.23 |
| **Validation** | 1.95 | 0.38 | 2.45 |
| **Test** | **1.93** | **0.39** | **2.42** |

✅ **Test MAE = 1.93**: Predictions are off by **1.93 corners** on average

**Currently my Model has overfit, I am still improving data and model configuration**



### Usual Error Distribution

```
Errors < 1 corner:    46%
Errors < 1.5 corners: 55%
Errors < 2 corners:   68%
Errors < 3 corners:   82%
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



## Key Findings

### What Works Well ✅
- Consistent teams → Better predictions (MAE ~1.9)
- Top leagues → More data = Better accuracy
- Mid-season matches → More historical data
- Matches where teams had low variance and low anomalies

### Challenges ⚠️
- Inconsistent teams → Higher error (MAE ~2.3)
- Early season → Limited historical data
- ***uncertainty***




---

## License

Educational purposes only. Not financial advice.

