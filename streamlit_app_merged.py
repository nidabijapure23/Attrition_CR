import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import requests

# Set page config
st.set_page_config(
    page_title="Attrition Prediction App",
    page_icon="📊",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.title("📊 Attrition Prediction")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app predicts employee attrition using machine learning. Upload your employee data to:
    - Predict attrition risk
    - Analyze risk factors
    - Get tenure predictions
    - Download detailed results
    """)
    
    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("""
    1. Upload your employee CSV file
    2. Click 'Predict' to run the analysis
    3. View the results and download predictions
    """)
    
    st.markdown("---")
    st.markdown("### Risk Levels")
    st.markdown("""
    - **Severe**: ≥ 90% probability
    - **More Likely**: ≥ 80% probability
    - **Intermediate**: ≥ 64% probability
    - **Mild**: ≥ 50% probability
    - **Minimal**: < 50% probability
    """)

# Load model
try:
    pipeline = joblib.load('CR_xgboost_model.pkl')
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Feature lists and category levels
ordinal_features = ['Education', 'Overall Manager Rating', 'Employee Category', 'Management Level']
nominal_features = ['Gender', 'Work Shift', 'Employee Type', 'Time Type', 'Job Profile']
numerical_features = ['Age', 'Years in Position', 'Years with Current Manager','Tenure',
                     'Years since Last Promotion', 'Total Base Pay - Amount', 'Last Base Pay Increase - Percent',
                     'Scheduled Weekly Hours', 'CF_LRV_company worked count']

# Category levels
education_levels = ['High School', 'GED', 'Associates', 'Bachelors', 'Masters', 'Doctorate', '']
rating_levels = ['Not Meeting Expectations', 'Below Expectations', 'Meets Expectations',
                 'Meeting Expectations', 'Exceeds Expectation', '']
employee_category_levels = ['Agents', 'Office', 'TL', 'MC', 'Management', '']
management_levels = ['Staff', 'Middle Management', 'Senior Executives', '']

# Preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('encoder', OrdinalEncoder(categories=[education_levels, rating_levels, employee_category_levels, management_levels],
                              handle_unknown='use_encoded_value', unknown_value=-1))
])

nominal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_features),
    ('ord', ordinal_transformer, ordinal_features),
    ('nom', nominal_transformer, nominal_features)
])

# Risk bucketing function
def bucketize_risk(prob):
    if prob >= 0.9:
        return 'Severe'
    elif prob >= 0.8:
        return 'More Likely'
    elif prob >= 0.64:
        return 'Intermediate Risk'
    elif prob >= 0.5:
        return 'Mild Risk'
    else:
        return 'Minimal Risk'

def calculate_tenure(row):
    current_date = pd.to_datetime('2025-04-21')
    if pd.notna(row.get('Termination Date - All')):
        return (row['Termination Date - All'] - row['Hire Date']).days / 30
    else:
        return (current_date - row['Hire Date']).days / 30

def process_predictions(df):
    # print("Starting predictions...")
    # print("Input DataFrame shape:", df.shape)
    # print("Available columns:", df.columns.tolist())
    
    # --- Feature engineering: replicate your notebook logic here ---
    if 'Hire Date' in df.columns:
        df['Hire Date'] = pd.to_datetime(df['Hire Date'], format='%d-%m-%Y', errors='coerce')
    if 'Termination Date - All' in df.columns:
        df['Termination Date - All'] = pd.to_datetime(df['Termination Date - All'], format='%d-%m-%Y', errors='coerce')
    
    # Calculate tenure
    current_date = pd.to_datetime('2025-04-21')
    def calculate_tenure(row):
        if pd.notna(row.get('Termination Date - All')):
            return (row['Termination Date - All'] - row['Hire Date']).days / 30
        else:
            return (current_date - row['Hire Date']).days / 30
    
    if 'Tenure' not in df.columns and 'Hire Date' in df.columns:
        df['Tenure'] = df.apply(calculate_tenure, axis=1)
    
    # print("After tenure calculation, DataFrame shape:", df.shape)
    
    # Prepare features for prediction
    feature_cols = numerical_features + ordinal_features + nominal_features
    # print("Required features:", feature_cols)
    # print("Missing features:", [col for col in feature_cols if col not in df.columns])
    
    X = df[feature_cols].copy()
    # print("Feature matrix shape:", X.shape)
    
    # Transform features using the pipeline's preprocessor
    try:
        X_transformed = pipeline.named_steps['preprocessor'].transform(X)
        # print("Transformed features shape:", X_transformed.shape)
        
        # Get predictions
        proba = pipeline.named_steps['classifier'].predict_proba(X_transformed)[:, 1]
        # print("Prediction probabilities shape:", proba.shape)
        # print("Sample probabilities:", proba[:5])
        
        df['Attrition Probability'] = proba
        df['Attrition Prediction'] = np.where(proba >= 0.5, 'Left', 'Stayed')
        # df['Attrition Prediction'] = pipeline.named_steps['classifier'].predict(X_transformed)
        df['Risk Level'] = df['Attrition Probability'].apply(bucketize_risk)
        
        # print("Predictions completed. Sample results:")
        # print(df[['Attrition Probability', 'Attrition Prediction', 'Risk Level']].head())
        
    except Exception as e:
        print("Error during prediction:", str(e))
        raise e
    
    # Calculate tenure prediction
    if 'Tenure' not in df.columns:
        df['Predicted Tenure'] = tenure_pipeline.predict(X)
    else:
        df['Predicted Tenure'] = df['Tenure']
    
    # Calculate SHAP values for feature importance
    try:
        explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])
        shap_values = explainer.shap_values(X_transformed)
        nom_features = pipeline.named_steps['preprocessor'].named_transformers_['nom'].named_steps['encoder'].get_feature_names_out(nominal_features)
        all_features = numerical_features + ordinal_features + list(nom_features)
        triggers = []
        for i in range(X.shape[0]):
            if df.iloc[i]['Attrition Probability'] >= 0.4:
                shap_vals = shap_values[i]
                top_indices = np.argsort(np.abs(shap_vals))[-5:][::-1]
                triggers.append(', '.join([all_features[j] for j in top_indices]))
            else:
                triggers.append('')
        df['Triggers'] = triggers
    except Exception as e:
        print('SHAP error:', e)
        df['Triggers'] = ''
    
    # Create Actual Status column
    def get_actual_status(row):
        if str(row.get('Active Status')).strip().lower() == 'yes':
            return 'Yes'
        if 'Termination Date - All' in row and pd.notna(row.get('Termination Date - All')):
            return 'No'
        return 'No'
    df['Actual Status'] = df.apply(get_actual_status, axis=1)
    
    # print("Final DataFrame shape:", df.shape)
    # print("Final columns:", df.columns.tolist())
    
    return df

# Main content
st.title("📊 Attrition Prediction App")

# File upload
uploaded_file = st.file_uploader("Upload Employee CSV", type="csv")

if uploaded_file:
    st.session_state.uploaded_file = uploaded_file
    st.write("File uploaded. Click Predict to run analysis.")
    if st.button("Predict"):
        with st.spinner('Running prediction...'):
            # Read and process the file
            df = pd.read_csv(uploaded_file)
            
            # Process predictions
            df = process_predictions(df)
            
            # Display summary
            st.subheader("📈 Prediction Summary")
            
            # Calculate summary statistics
            total_active = (df['Attrition Prediction'] == 'Stayed').sum()
            total_inactive = (df['Attrition Prediction'] == 'Left').sum()
            risk_level_counts = df[df['Attrition Prediction'] == 'Left'].groupby('Risk Level').size().to_dict()
            
            # Create summary table
            summary_data = {
                "Metric": [
                    "Total Active Predictions",
                    "Total Inactive Predictions"
                ],
                "Count": [
                    total_active,
                    total_inactive
                ]
            }
            
            # Add risk level counts to summary
            for risk_level, count in risk_level_counts.items():
                summary_data["Metric"].append(f"{risk_level} Count")
                summary_data["Count"].append(count)
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)

            # # Display risk level distribution
            # st.subheader("🎯 Risk Level Distribution")
            # risk_data = pd.DataFrame({
            #     "Risk Level": list(risk_level_counts.keys()),
            #     "Count": list(risk_level_counts.values())
            # })
            # st.bar_chart(risk_data.set_index("Risk Level"))

            # Display false positives (active employees predicted as attrite)
            st.subheader("⚠️ Active Employees Predicted as at Risk")
            false_positives = df[(df['Attrition Prediction'] == 'Left') & (df['Actual Status'] == 'Yes')]
            if not false_positives.empty:
                display_cols = ['Employee ID', 'Attrition Prediction', 'Risk Level', 'Attrition Probability', 'Triggers', 'Predicted Tenure']
                st.dataframe(false_positives[display_cols], use_container_width=True)
            else:
                st.success("No active employees predicted as at risk!")

            # Save and provide download for all predictions
            st.subheader("📥 Download Predictions")
            all_predictions = df[['Employee ID', 'Attrition Prediction', 'Risk Level', 'Attrition Probability', 'Triggers', 'Predicted Tenure', 'Actual Status']]
            all_predictions.to_csv('all_predictions.csv', index=False)
            with open('all_predictions.csv', 'rb') as f:
                st.download_button(
                    label="Download All Predictions",
                    data=f,
                    file_name='all_predictions.csv',
                    mime='text/csv',
                    help="Click to download the complete predictions file"
                )
# Tenure regressor pipeline
tenure_numerical_features = [f for f in numerical_features if f != 'Tenure']
tenure_preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, tenure_numerical_features),
    ('ord', ordinal_transformer, ordinal_features),
    ('nom', nominal_transformer, nominal_features)
])
tenure_pipeline = Pipeline(steps=[
    ('preprocessor', tenure_preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
]) 