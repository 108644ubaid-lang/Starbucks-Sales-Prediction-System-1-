import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Starbucks ML Prediction")
st.title("Starbucks Machine Learning Prediction")
st.write("Decision Tree Classification")

# Keep these model files in the same GitHub repository folder as app.py.
required_files = [
    "decision_tree.pkl",
    "imputer.pkl",
    "encoder.pkl",
    "features.pkl",
    "target.pkl",
]

missing_files = [file for file in required_files if not os.path.exists(file)]
if missing_files:
    st.error(
        "Model files are missing: " + ", ".join(missing_files)
        + ". Please add all five .pkl files to the project repository."
    )
    st.stop()

model = joblib.load("decision_tree.pkl")
imputer = joblib.load("imputer.pkl")
encoder = joblib.load("encoder.pkl")
features = joblib.load("features.pkl")
target = joblib.load("target.pkl")

st.write("Prediction target:", target)
st.subheader("Enter feature values")

uploaded = st.file_uploader(
    "Optional: Upload a CSV containing input features",
    type=["csv"]
)

if uploaded is not None:
    raw = pd.read_csv(uploaded)
    raw.columns = raw.columns.astype(str).str.strip()
    missing = [f for f in features if f not in raw.columns]

    if missing:
        st.error("Missing required feature columns: " + ", ".join(missing))
        st.info(
            "The uploaded CSV must have exactly the feature columns used "
            "when the model was trained. Upload the model's original "
            "preprocessed feature dataset."
        )
    else:
        data = raw[features].copy()
        data = data.replace([float("inf"), float("-inf")], float("nan"))
        data = pd.DataFrame(imputer.transform(data), columns=features)
        result = encoder.inverse_transform(model.predict(data))
        output = raw.copy()
        output["Prediction"] = result
        st.dataframe(output)
        st.download_button(
            "Download Predictions",
            output.to_csv(index=False),
            "predictions.csv",
            "text/csv"
        )
else:
    st.write("Enter values for one prediction.")
    values = {}
    for feature in features:
        values[feature] = st.number_input(
            feature, value=0.0, key=str(feature)
        )

    if st.button("Predict"):
        data = pd.DataFrame([values], columns=features)
        data = data.replace([float("inf"), float("-inf")], float("nan"))
        data = pd.DataFrame(imputer.transform(data), columns=features)
        result = encoder.inverse_transform(model.predict(data))[0]
        st.success("Predicted category: " + str(result))
