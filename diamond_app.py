# import required libraries 
import streamlit as st 
import pandas as pd 
import joblib 

# Load the saved model, scaler and encoder
model = joblib.load('random_forest_model.pkl') 
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

# App title 
st.title('Diamond Price Prediction App') 
st.write('Enter the features of the diamond to predict its price.')

# input fields for diamond features with descriptions 
carat = st.number_input(
    "Carat Weight (Weight of the diamond in carats)",
    min_value=0.0, max_value=5.0, step=0.01, format="%.2f" 
)
cut = st.selectbox(
    'Cut Quality (Quality of the Cut: Fair, Good, Very Good, Premium, Ideal)',
    ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
)
color = st.selectbox(
    'Color Grade (Diamond color: 3 (worst) to D (best))',
    ['J', 'I', 'H', 'G', 'F', 'E', 'D']
)
clarity = st.selectbox(
    "Clarity Grade (Diamond clarity: I1 (worst) to IF (best))",
    ['I1', 'S12', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS2', 'IF']
)
depth = st.number_input(
    "Depth Percentage (Total depth percentage of the diamond)",
    min_value=0.0, max_value=100.0, step=0.1, format='%.1f' 
) 
table = st.number_input(
    "Table Percentage (Width of the top of the diamond relatives to the widest point)",
    min_value=0.0, max_value=100.0, step=0.1, format="%.1f"
)
x = st.number_input(
    "Length (mm) (Length of the diamond in millimeters)",
    min_value= 0.0, max_value=100.0, step=0.1, format='%.1f'
)
y = st.number_input(
    "Width (mm) (Width of the diamond in millimeters)",
    min_value=0.0, max_value=100.0, step=0.1, format='%.1f'
) 
z = st.number_input(
    "Depth (mm) (Depth of the diamond in millimeters)",
    min_value=0.0, max_value=100.0, step=0.1, format='%.1f'
)

# Predict the price 
if st.button("Predict Price"):
    # input the data
    input_data = pd.DataFrame({
        'carat': [carat],
        'cut': [cut],
        'color': [color],
        'clarity': [clarity],
        'depth': [depth],
        'table': [table],
        'x': [x],
        'y': [y],
        'z': [z]
    }
) 

try:
    input_data = pd.DataFrame({
        'carat': [carat],
        'cut': [cut],
        'color': [color],
        'clarity': [clarity],
        'depth': [depth],
        'table': [table],
        'x': [x],
        'y': [y],
        'z': [z]
    })
    print("DataFrame created successfully")  # Debug message
except Exception as e:
    print(f"Error creating DataFrame: {e}") 
    
# Separate categorical and numerical features
input_data_cat = input_data[['cut', 'color', 'clarity']] 
input_data_num = input_data.drop(columns=['cut', 'color', 'clarity']) 

# Applying transforming on numerical columns 
input_data_num_transformed = pd.DataFrame(
    scaler.transform(input_data_num),
    columns = scaler.get_feature_names_out(),
    index=input_data_num.index
) 

# Applying transformation on categorical column 
input_data_cat_transformed = pd.DataFrame(
    encoder.transform(input_data_cat),
    columns=encoder.get_feature_names_out(['cut', 'color', 'clarity']),
    index=input_data_cat.index
) 

# Concatenating numerical and categorical transformed data
processed_data = pd.concat([input_data_num_transformed, input_data_cat_transformed], axis=1) 

# Predict using the loaded model 
prediction = model.predict(processed_data)

# Display the result 
st.success(f"The predicted price of the diamond is: ${prediction[0]:,.2f}") 
