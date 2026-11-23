
import streamlit as st
import pandas as pd
import joblib
import json

# Page configuration
st.set_page_config(
    page_title="Energy Consumption Prediction",
    page_icon="⚡",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("energy_model.pkl")

model = load_model()

# Load feature information
with open("feature_info.json", "r") as f:
    info = json.load(f)

# Application heading
st.title("⚡ Energy Consumption Prediction")
st.subheader("Machine Learning Based Prediction System")

st.write(
    "Predict building energy consumption "
    "using machine learning."
)

st.divider()

st.header("Enter Building Details")

user_input = {}

# Create input fields
with st.form("prediction_form"):

    for col in info["columns"]:

        label = col.replace("_", " ").title()
        default = info["defaults"][col]

        if col in info["numeric"]:

            user_input[col] = st.number_input(
                label,
                value=float(default)
            )

        else:

            options = info["categories"].get(
                col, [default]
            )

            user_input[col] = st.selectbox(
                label,
                options=options,
                index=(
                    options.index(default)
                    if default in options else 0
                )
            )

    submitted = st.form_submit_button(
        "⚡ Predict Energy Consumption",
        type="primary",
        use_container_width=True
    )

# Generate prediction
if submitted:

    input_df = pd.DataFrame(
        [user_input],
        columns=info["columns"]
    )

    prediction = model.predict(input_df)[0]

    st.success("Prediction completed!")

    st.metric(
        "Predicted Energy Consumption",
        f"{prediction:.2f}"
    )

    st.caption(
        "Result is expressed in the energy "
        "consumption units used by the dataset."
    )

    st.write("Input details:")
    st.dataframe(input_df, use_container_width=True)

st.divider()

st.markdown(
    "**Project:** Energy Consumption Prediction  \n"
    "**Algorithm:** Gradient Boosting Regression  \n"
    "**SDG 7:** Affordable and Clean Energy  \n"
    "**SDG 13:** Climate Action"
)
