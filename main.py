import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

model = joblib.load("trained_model.joblib")
scaler = joblib.load("scaler.joblib")

final_features = [
    ("Cr", "Chromium"),
    ("C", "Carbon"),
    ("Mo", "Molybdenum"),
    ("Mn", "Manganese"),
    ("Ni", "Nickel"),
    ("Si", "Silicon"),
    ("TT", "Tempering temperature"),
    ("NT", "Normalizing temperature"),
    ("RedRatio", "Reduction Ratio"),
    ("THT", "Through hardening temperature"),
    ("TCr", "Cooling rate for Tempering"),
    ("THQCr", "Cooling rate for Through hardening"),
    ("Tt", "Tempering Time"),
    ("CT", "Carburization temperature"),
    ("Dt", "Diffusion Time"),
]

st.set_page_config(page_title='Material Fatigue Classifier', layout='wide')
st.title("🌟 Material Fatigue Class Prediction")
st.write("Enter material properties to predict its fatigue class:")

input_vals = []
cols = st.columns(3)

for i, (abb, full_name) in enumerate(final_features):
    with cols[i % 3]:
        val = st.number_input(f'{abb} - {full_name}', step=0.01,value=0.0)
        input_vals.append(val)

if st.button("Predict"):
    sample = np.array([input_vals])

    sample_scaled = scaler.transform(sample)

    predicted_class = model.predict(sample_scaled)[0]

    if predicted_class == 0:
        result = "The Material is Weak with Fatigue Strength (< 400 MPa)"
        color = "red"
        icon = "❌"
    elif predicted_class == 1:
        result = "The Material is Moderate in Strength with Fatigue Strength (400-600 MPa)"
        color = "#ff8c00"
        icon = "⚠"
    else:
        result = "The Material is Strong with Fatigue Strength (> 600 MPa)"
        color = "green"
        icon = "✅"

    st.markdown(
        f"""<h2 style='color:{color}; text-align: center; font-size: 2.5em; margin-bottom: 20px'>{icon} Material is {result}</h2>""",
        unsafe_allow_html=True
    )

if st.button("View Feature Importances"):
    """Compute and plot SHAP feature importances """

    explainer = shap.TreeExplainer(model)
    sample = scaler.transform([input_vals])

    shap_values = explainer.shap_values(sample)

    mean_vals_per_classes = np.abs(shap_values).mean(0)

    mean_vals = mean_vals_per_classes.mean(1)
    feature_importances = {
        "feature": [f[0] for f in final_features],
        "mean_abs_shap": mean_vals
    }
    feature_importances_sorted = sorted(
        zip(feature_importances["feature"], feature_importances["mean_abs_shap"]),
        key=lambda x: x[1],
        reverse=True
    )

    features, scores = zip(*[(dict(final_features)[f], s) for f, s in feature_importances_sorted])

    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=scores, y=features, color='#ff7f50')
    ax.set_title('Feature Importances Based on SHAP')
    ax.set_xlabel('Mean |SHAP| Score')
    ax.set_ylabel('Features')
    ax.grid(visible=False)
    ax.set_facecolor("#fff5cc")
    fig.patch.set_facecolor("#fff5cc")

    st.pyplot(fig)
