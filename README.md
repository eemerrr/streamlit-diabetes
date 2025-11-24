# Diabetes Prediction (Streamlit)

Streamlit app to estimate diabetes status using the provided `best_model.pkl` XGBoost classifier. The app collects the BRFSS-derived features (excluding `DIABETE4` and `DIABTYPE`) and returns the probability of a positive diabetes diagnosis.

## Requirements
- Python 3.10+ (tested with 3.13)
- `best_model.pkl` in the project root

Install dependencies:
```bash
py -m pip install -r requirements.txt
```

## Running the app
```bash
py -m streamlit run app.py
```
If `streamlit.exe` is blocked, using `py -m streamlit` avoids the executable.

## Inputs used by the model
- Binary (yes/no): `_RFHYPE6`, `_RFCHOL3`, `_CHOLCH3`, `SMOKE100`, `CVDSTRK3`, `_MICHD`, `EXERANY2`, `MEDCOST1`, `DIFFWALK`, `_SEX`
- Ordinal / categorical codes: `PRIMINS1`, `GENHLTH (1=Excellent..5=Poor)`, `_AGEG5YR`, `_EDUCAG`, `_INCOMG1`
- Numeric: `_BMI5` (BMI × 10), `MENTHLTH` (0–30 days), `PHYSHLTH` (0–30 days)

The UI converts standard BMI to `_BMI5` automatically. The displayed encoded values can be inspected in the “View encoded feature values” expander after submitting. Already-installed xgboost may emit a warning about loading an older pickle; it still loads and predicts.***
