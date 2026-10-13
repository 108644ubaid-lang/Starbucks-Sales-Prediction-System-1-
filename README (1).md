# Starbucks Machine Learning Prediction

A Streamlit application for Starbucks menu category prediction using a Decision Tree model.

## Files required

The following trained model artifacts are required in the same folder as `app.py`:

- `decision_tree.pkl`
- `imputer.pkl`
- `encoder.pkl`
- `features.pkl`
- `target.pkl`

**Important:** These five `.pkl` files were not included in the files available when this ZIP was prepared. Add the exact files created during model training before running or deploying the app. Without them, the prediction app cannot start.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

1. Upload this project folder to a GitHub repository.
2. Add the five required `.pkl` files to the repository root.
3. Open Streamlit Community Cloud and connect the GitHub repository.
4. Select `app.py` as the main file and deploy.

`Correct CSV File - Fixed.csv` is included as a data file. It is not guaranteed to match the model's saved `features.pkl` feature list; the uploaded prediction CSV must use the exact feature columns used during training.
