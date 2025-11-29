import numpy as np
import pickle
import streamlit as st 

# Load model
loaded_model = pickle.load(open(
    "C:/Users/ShivamShubham/Desktop/medical insurance cost prediction/insurance_model.sav",
    "rb"
))

# Prediction function
def insurance_charge_prediction(input_data):

    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    input_data_reshape = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshape)
    return f"The insurance cost is USD : {prediction[0]:.2f}"


def main():
    st.title("INSURANCE CHARGE PREDICTION")

    # User Inputs
    age = st.number_input("Age", min_value=0, max_value=120)
    bmi = st.number_input("BMI", min_value=0.0)
    children = st.number_input("Number of children", min_value=0, max_value=10)

    # Sex dropdown
    sex = st.selectbox("Sex", ("Male", "Female"))
    sex_encoded = 0 if sex == "Male" else 1

    # Smoker dropdown
    smoker = st.selectbox("Smoker", ("Yes", "No"))
    smoker_encoded = 0 if smoker == "Yes" else 1

    # Region dropdown
    region = st.selectbox(
        "Region",
        ("Southeast", "Southwest", "Northeast", "Northwest")
    )
    region_mapping = {
        "Southeast": 0,
        "Southwest": 1,
        "Northeast": 2,
        "Northwest": 3
    }
    region_encoded = region_mapping[region]

    cost = ""

    if st.button("Predict Charge"):
        input_values = [
            age,
            sex_encoded,
            bmi,
            children,
            smoker_encoded,
            region_encoded
        ]
        cost = insurance_charge_prediction(input_values)

    st.success(cost)


if __name__ == "__main__":
    main()
   