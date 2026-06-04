# Waste Prediction Dashboard

Predict waste load weights from historical collection data using machine learning. Features an interactive web dashboard for what-if scenario testing and data filtering.

## Project Structure

```
├── analysis/
│   └── explore_data.py       # Data exploration & profiling
├── app/
│   ├── app.py                # Flask web application
│   └── templates/
│       └── index.html        # Frontend dashboard UI
├── model/
│   ├── train_model.py        # Training pipeline
│   └── waste_predictor.pkl   # Trained model + preprocessor
├── preprocessing/
│   ├── preprocess_data.py    # Preprocessor creation script
│   └── waste_preprocessor.pkl
├── utils/
│   └── preprocessor.py       # WasteDataPreprocessor class
├── largeClean.csv            # Raw dataset (740k+ rows)
├── README.md
└── .gitignore
```

## Pipeline

1. **Explore** – `analysis/explore_data.py` profiles the raw dataset (distributions, missing values, outliers)
2. **Preprocess** – `utils/preprocessor.py` defines a `WasteDataPreprocessor` that extracts temporal features (`reportYear`, `reportMonth`, `reportDayOfWeek`, `loadHour`) and label-encodes categorical columns (`loadType`, `dropoffSite`, `routeType`, `routeNumber`)
3. **Train** – `model/train_model.py` trains a Random Forest Regressor on 100k samples (Test MAE ~3322, R² ~0.59) and serialises the model + preprocessor to `waste_predictor.pkl`
4. **Serve** – `app/app.py` serves the dashboard and prediction API

## Getting Started

### Requirements

- Python 3.10+
- pandas, numpy, scikit-learn, flask

### Setup

```bash
cd waste-regress
pip install pandas numpy scikit-learn flask

# Train the model (takes ~2 minutes)
python preprocessing/preprocess_data.py
python model/train_model.py

# Start the dashboard
python app/app.py
```

Open `http://localhost:5000` in your browser.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Renders the dashboard UI |
| `/predict` | POST | Predict waste weight from categorical + temporal inputs |
| `/filter-data` | POST | Query filtered records from the dataset |
| `/predict-filtered` | POST | Run predictions on filtered data for comparison charts |
| `/get-options` | GET | Available values for loadType, dropoffSite, routeType |
| `/model-info` | GET | Feature importance and model metadata |

### Prediction Request

```json
{
  "loadType": "GARBAGE COLLECTIONS",
  "dropoffSite": "TDS LANDFILL",
  "routeType": "GARBAGE COLLECTION",
  "loadHour": 12,
  "reportDayOfWeek": 1,
  "reportMonth": 6,
  "reportYear": 2020
}
```

Returns:

```json
{
  "prediction": 15330.75,
  "status": "success"
}
```

## Dataset

`largeClean.csv` contains 740,873 records of waste collection loads with columns:

- `reportDate` – Collection date
- `loadType` – Category of waste (GARBAGE COLLECTIONS, RECYCLING, ORGANICS, etc.)
- `loadTime` – Time of weighing
- `loadWeight` – Target variable (lbs)
- `dropoffSite` – Disposal facility
- `routeType` – Collection route classification
- `routeNumber` – Route identifier
- `loadID` – Unique load identifier
