import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="sasipriyank/tourist-model", filename="best_tour_model.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Tourisim Prediction App")
st.write("The Tourist Prediction App is an internal tool for Agent that predicts whether customers are buy the Package or not.")
st.write("Kindly enter the customer details to check whether they are likely to purchase or not.")

# Collect user input

NumberOfPersonVisiting = st.number_input("Number Of Person Visiting",min_value=1, value=1)
NumberOfChildrenVisiting= st.number_input("Number Of Children Visiting",min_value=0, value=0)
#DurationOfPitch = st.number_input("Duration Of Pitch",min_value=1, value=1)
Occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business","Large Business"])
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
Gender = st.selectbox("Gender", ["Male", "Female"])
MaritalStatus = st.selectbox("Marital Status", ["Married","Unmarried", "Single","Divorced"])
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "VP","AVP"])
#ProductPitched=  st.selectbox("Product pitched", ["Basic", "Standard","Deluxe"])
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
CityTier = st.selectbox("CityTier", ["1", "2","3"])
PreferredPropertyStar = st.selectbox("PreferredPropertyStar", ["1", "2","3","4","5"])
#PitchSatisfactionScore= st.selectbox("PitchSatisfactionScore",["1", "2","3","4","5"])
#NumberOfFollowups = st.number_input("Number Of Followups", min_value=0, value=1)
NumberOfTrips = st.number_input("Number Of Trips", min_value=0, value=1)
Passport = st.selectbox("have passport?", ["Yes", "No"])
OwnCar = st.selectbox("Is own car?", ["Yes", "No"])
MonthlyIncome = st.number_input("Monthly Salary", min_value=0, value=30000)

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'Occupation': Occupation,
    'Age': Age,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    #'DurationOfPitch': DurationOfPitch,
    'TypeofContact': TypeofContact,
    'Gender': Gender,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation,
    #'ProductPitched': ProductPitched,
    #'PitchSatisfactionScore': PitchSatisfactionScore,
    'CityTier': CityTier,
    #'NumberOfFollowups' : NumberOfFollowups,
    'NumberOfTrips' : NumberOfTrips,
    'PreferredPropertyStar': PreferredPropertyStar,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'Passport': 1 if Passport == "Yes" else 0,
    'MonthlyIncome': MonthlyIncome
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "purchased" if prediction == 1 else "not purchased"
    st.write(f"Based on the information provided, the customer is  {result} the new tourist package")
