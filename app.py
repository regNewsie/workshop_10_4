import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained model
with open('best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as label_encoder_file:
    label_encoder = pickle.load(label_encoder_file)

# Title of the app
st.title("Coffee Type Prediction")

# Sidebar inputs for user preferences
st.sidebar.header("User Preferences")

time_of_day = st.sidebar.selectbox("Time of Day", ['morning', 'afternoon', 'evening'])
coffee_strength = st.sidebar.selectbox("Coffee Strength", ['mild', 'regular', 'strong'])
sweetness_level = st.sidebar.selectbox("Sweetness Level", ['unsweetened', 'lightly sweetened', 'sweet'])
milk_type = st.sidebar.selectbox("Milk Type", ['none', 'regular', 'skim', 'almond'])
coffee_temperature = st.sidebar.selectbox("Coffee Temperature", ['hot', 'iced', 'cold brew'])
flavored_coffee = st.sidebar.selectbox("Flavored Coffee", ['yes', 'no'])
caffeine_tolerance = st.sidebar.selectbox("Caffeine Tolerance", ['low', 'medium', 'high'])
coffee_bean = st.sidebar.selectbox("Coffee Bean", ['Arabica', 'Robusta', 'blend'])
coffee_size = st.sidebar.selectbox("Coffee Size", ['small', 'medium', 'large'])
dietary_preferences = st.sidebar.selectbox("Dietary Preferences", ['none', 'vegan', 'lactose-intolerant'])

# Encoding the inputs manually (same encoding as in your training data)
input_data = pd.DataFrame({
    'Token_0': [time_of_day],
    'Token_1': [coffee_strength],
    'Token_2': [sweetness_level],
    'Token_3': [milk_type],
    'Token_4': [coffee_temperature],
    'Token_5': [flavored_coffee],
    'Token_6': [caffeine_tolerance],
    'Token_7': [coffee_bean],
    'Token_8': [coffee_size],
    'Token_9': [dietary_preferences]
})

# One-hot encode the input data (ensure it matches the training data)
input_encoded = pd.get_dummies(input_data, sparse=False)

# Align columns with the training data (required columns)
required_columns = model.feature_names_in_  # Get the feature columns from the model
for col in required_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[required_columns]

# Make the prediction
prediction = model.predict(input_encoded)[0]

# Reverse the label encoding (map the prediction back to the coffee type)
coffee_type = label_encoder.inverse_transform([prediction])[0]

# Display the prediction
st.subheader(f"Recommended Coffee: {coffee_type}")