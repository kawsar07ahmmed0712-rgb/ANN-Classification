import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# -----------------------------
# Page config + small styling
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
)

st.title("📉 Customer Churn Prediction")
st.caption("Enter customer details to estimate churn probability using a trained TensorFlow model.")

MODEL_PATH = "model.keras"
GENDER_ENCODER_PATH = "label_encoder_gender.pkl"
GEO_ENCODER_PATH = "onehot_encoder_geo.pkl"
SCALER_PATH = "scaler.pkl"


# -----------------------------
# Cached asset loading
# -----------------------------
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(GENDER_ENCODER_PATH, "rb") as f:
        label_encoder_gender = pickle.load(f)

    with open(GEO_ENCODER_PATH, "rb") as f:
        onehot_encoder_geo = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    return model, label_encoder_gender, onehot_encoder_geo, scaler


def safe_number(value, min_val=None, max_val=None):
    """Clamp numeric values to avoid weird edge cases."""
    if value is None:
        return 0.0
    v = float(value)
    if min_val is not None:
        v = max(v, min_val)
    if max_val is not None:
        v = min(v, max_val)
    return v


def build_features(
    geography: str,
    gender: str,
    age: int,
    balance: float,
    credit_score: float,
    estimated_salary: float,
    tenure: int,
    num_of_products: int,
    has_cr_card: int,
    is_active_member: int,
    label_encoder_gender,
    onehot_encoder_geo,
):
    # Encode gender
    gender_enc = int(label_encoder_gender.transform([gender])[0])

    # Base numeric features
    X = pd.DataFrame(
        {
            "CreditScore": [safe_number(credit_score, 0, 1000)],
            "Gender": [gender_enc],
            "Age": [int(age)],
            "Tenure": [int(tenure)],
            "Balance": [safe_number(balance, 0)],
            "NumOfProducts": [int(num_of_products)],
            "HasCrCard": [int(has_cr_card)],
            "IsActiveMember": [int(is_active_member)],
            "EstimatedSalary": [safe_number(estimated_salary, 0)],
        }
    )

    # One-hot encode Geography
    geo_encoded = onehot_encoder_geo.transform([[geography]])
    if hasattr(geo_encoded, "toarray"):
        geo_encoded = geo_encoded.toarray()

    geo_cols = onehot_encoder_geo.get_feature_names_out(["Geography"])
    geo_df = pd.DataFrame(geo_encoded, columns=geo_cols)

    # Combine
    X = pd.concat([X.reset_index(drop=True), geo_df], axis=1)
    return X


def align_columns_for_scaler(X: pd.DataFrame, scaler) -> pd.DataFrame:
    """
    Protects you from column order mismatch.
    If scaler was fit on a pandas DataFrame, it may have feature_names_in_.
    Otherwise we assume current order is correct.
    """
    if hasattr(scaler, "feature_names_in_"):
        needed = list(scaler.feature_names_in_)
        missing = [c for c in needed if c not in X.columns]
        extra = [c for c in X.columns if c not in needed]

        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        if extra:
            # Not fatal, but indicates mismatch between training and app pipeline
            X = X[needed]  # drop extras by reindexing

        X = X.reindex(columns=needed)

    return X


# -----------------------------
# Load assets safely
# -----------------------------
try:
    model, label_encoder_gender, onehot_encoder_geo, scaler = load_assets()
except Exception as e:
    st.error("Failed to load model or preprocessing assets. Check file paths and compatibility.")
    st.exception(e)
    st.stop()


# -----------------------------
# Sidebar inputs
# -----------------------------
with st.sidebar:
    st.header("Inputs")

    geography = st.selectbox(
        "Geography",
        options=list(onehot_encoder_geo.categories_[0]),
        help="Country/region used for one-hot encoding."
    )

    gender = st.selectbox(
        "Gender",
        options=list(label_encoder_gender.classes_),
        help="Will be label-encoded using the saved encoder."
    )

    age = st.slider("Age", 18, 92, 35)
    tenure = st.slider("Tenure (years)", 0, 10, 3)
    num_of_products = st.slider("Number of Products", 1, 4, 1)

    credit_score = st.number_input("Credit Score", min_value=0.0, max_value=1000.0, value=650.0, step=1.0)
    balance = st.number_input("Balance", min_value=0.0, value=50000.0, step=100.0)
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=60000.0, step=100.0)

    has_cr_card = st.radio("Has Credit Card?", options=[0, 1], horizontal=True)
    is_active_member = st.radio("Is Active Member?", options=[0, 1], horizontal=True)

    st.divider()
    threshold = st.slider("Decision Threshold", 0.05, 0.95, 0.50, 0.01)
    show_debug = st.checkbox("Show debug (features table)", value=False)


# -----------------------------
# Main area: prediction
# -----------------------------
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Prediction")

    try:
        X = build_features(
            geography=geography,
            gender=gender,
            age=age,
            balance=balance,
            credit_score=credit_score,
            estimated_salary=estimated_salary,
            tenure=tenure,
            num_of_products=num_of_products,
            has_cr_card=has_cr_card,
            is_active_member=is_active_member,
            label_encoder_gender=label_encoder_gender,
            onehot_encoder_geo=onehot_encoder_geo,
        )

        X = align_columns_for_scaler(X, scaler)
        X_scaled = scaler.transform(X)

        # Model inference
        pred = model.predict(X_scaled, verbose=0)
        churn_proba = float(pred[0][0])

        st.metric("Churn Probability", f"{churn_proba:.2%}")

        st.progress(min(max(churn_proba, 0.0), 1.0))

        if churn_proba >= threshold:
            st.error(f"Likely to churn (≥ {threshold:.2f})")
        else:
            st.success(f"Not likely to churn (< {threshold:.2f})")

    except Exception as e:
        st.error("Prediction failed due to input/preprocessing mismatch.")
        st.exception(e)

with col2:
    st.subheader("What this means")
    st.write(
        """
        - This score is the model’s estimated probability of churn.
        - Use the threshold slider to change the decision boundary.
        - If your training pipeline changes, ensure the **saved encoders/scaler**
          match the app’s preprocessing exactly.
        """
    )

    st.info(
        "Tip: If you see column mismatch errors, your scaler may have been fit on a different set/order of features."
    )

if "X" in locals() and show_debug:
    st.subheader("Debug: Model Features")
    st.dataframe(X, use_container_width=True)