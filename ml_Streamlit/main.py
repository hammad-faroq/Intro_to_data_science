import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Customer Churn Machine Learning Model Workflow", layout="wide")

# === Utility Functions ===
@st.cache_resource
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    return df

def preprocess_data(df):
    df = df.copy()
    df = df.drop('customerID', axis=1)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'Churn':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})
    return df, label_encoders

def get_model(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    return model, acc, X.columns.tolist(), X_test, y_test

def eda_charts(data):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(x='Churn', data=data, ax=ax[0])
    ax[0].set_title("Churn Count")
    sns.histplot(data['tenure'], bins=30, kde=True, ax=ax[1])
    ax[1].set_title("Distribution of Tenure")
    st.pyplot(fig)
    st.write("**Churn Proportion:**")
    st.write(data['Churn'].value_counts(normalize=True).rename({0:'No',1:'Yes'}).to_frame("Proportion"))

    st.write("**Average Monthly Charges by Churn:**")
    st.bar_chart(data.groupby('Churn')['MonthlyCharges'].mean().rename({0:'No',1:'Yes'}))

    st.write("**Correlation Matrix:**")
    corr = data.corr()
    fig2, ax2 = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax2)
    st.pyplot(fig2)

# === App Sections ===
def section_data_loading(df):
    st.header("Step 1: Data Loading")
    st.info("**What happens in this step?**\n\n- The original Telco Customer Churn dataset is loaded from CSV file.\n- You can view the first few rows to understand the data.")
    st.write("**Sample of the raw data:**")
    st.dataframe(df.head(10))

def section_data_preprocessing(df, df_processed):
    st.header("Step 2: Data Preprocessing")
    st.info(
        "**What happens in this step?**\n"
        "- Remove unnecessary columns (like `customerID`).\n"
        "- Convert `TotalCharges` to numeric and handle missing values.\n"
        "- Encode categorical variables (convert text to numbers).\n"
        "- Encode the target `Churn` to 0/1 for modeling."
    )
    st.write("**Sample after preprocessing:**")
    st.dataframe(df_processed.head(10))

def section_eda(df_processed):
    st.header("Step 3: Exploratory Data Analysis (EDA)")
    st.info(
        "**What happens in this step?**\n"
        "- Visualize distribution of target and features.\n"
        "- Check correlations and trends.\n"
        "- Understand which features may affect churn."
    )
    eda_charts(df_processed)

def section_model_training(df_processed, model, accuracy, feature_names, X_test, y_test):
    st.header("Step 4: Model Training & Evaluation")
    st.info(
        "**What happens in this step?**\n"
        "- Split data into training and testing sets.\n"
        "- Train a Random Forest classifier.\n"
        "- Evaluate model accuracy."
    )
    st.write(f"**Model Accuracy on Test Set:** `{accuracy:.2%}`")
    # Feature importance plot
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    st.write("**Top 10 Feature Importances:**")
    st.bar_chart(feat_imp[:10])

    # Show Confusion Matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Stay','Churn'])
    disp.plot(ax=ax)
    st.pyplot(fig)

def section_prediction(model, label_encoders):
    st.header("Step 5: Make a Prediction")
    st.info(
        "**What happens in this step?**\n"
        "- Enter details for a new customer.\n"
        "- The model predicts if this customer will churn."
    )
    st.write("**Enter customer details below:**")
    # Input form
    with st.form("prediction_form"):
        gender = st.selectbox('Gender', label_encoders['gender'].classes_)
        senior = st.selectbox('Senior Citizen', [0, 1])
        partner = st.selectbox('Partner', label_encoders['Partner'].classes_)
        dependents = st.selectbox('Dependents', label_encoders['Dependents'].classes_)
        tenure = st.slider('Tenure (months)', 0, 72, 12)
        phone = st.selectbox('Phone Service', label_encoders['PhoneService'].classes_)
        multiple = st.selectbox('Multiple Lines', label_encoders['MultipleLines'].classes_)
        internet = st.selectbox('Internet Service', label_encoders['InternetService'].classes_)
        online_sec = st.selectbox('Online Security', label_encoders['OnlineSecurity'].classes_)
        online_bak = st.selectbox('Online Backup', label_encoders['OnlineBackup'].classes_)
        device = st.selectbox('Device Protection', label_encoders['DeviceProtection'].classes_)
        tech = st.selectbox('Tech Support', label_encoders['TechSupport'].classes_)
        str_tv = st.selectbox('Streaming TV', label_encoders['StreamingTV'].classes_)
        str_mov = st.selectbox('Streaming Movies', label_encoders['StreamingMovies'].classes_)
        contract = st.selectbox('Contract', label_encoders['Contract'].classes_)
        paperless = st.selectbox('Paperless Billing', label_encoders['PaperlessBilling'].classes_)
        payment = st.selectbox('Payment Method', label_encoders['PaymentMethod'].classes_)
        monthly = st.number_input('Monthly Charges', min_value=0.0, max_value=200.0, value=70.0)
        total = st.number_input('Total Charges', min_value=0.0, max_value=10000.0, value=2000.0)
        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        input_dict = {
            'gender': gender,
            'SeniorCitizen': senior,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone,
            'MultipleLines': multiple,
            'InternetService': internet,
            'OnlineSecurity': online_sec,
            'OnlineBackup': online_bak,
            'DeviceProtection': device,
            'TechSupport': tech,
            'StreamingTV': str_tv,
            'StreamingMovies': str_mov,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment,
            'MonthlyCharges': monthly,
            'TotalCharges': total
        }
        input_df = pd.DataFrame([input_dict])
        # Encode input
        for col in input_df.columns:
            if col in label_encoders:
                le = label_encoders[col]
                input_df[col] = le.transform([input_df[col][0]])
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        st.markdown(f"**Churn Probability:** `{proba:.2%}`")
        if prediction == 1:
            st.error("⚠️ This customer is likely to **CHURN**.")
        else:
            st.success("✅ This customer is likely to **STAY**.")

# === Main App ===
def main():
    st.title("📱 Customer Churn Full ML Workflow")

    menu = [
        "Data Loading",
        "Data Preprocessing",
        "EDA (Data Analysis)",
        "Model Training",
        "Prediction",
        "About Project"
    ]
    choice = st.sidebar.radio("Go to Step:", menu)

    # Load and preprocess data
    df = load_data()
    df_processed, label_encoders = preprocess_data(df)
    model, accuracy, feature_names, X_test, y_test = get_model(df_processed)

    if choice == "Data Loading":
        section_data_loading(df)
    elif choice == "Data Preprocessing":
        section_data_preprocessing(df, df_processed)
    elif choice == "EDA (Data Analysis)":
        section_eda(df_processed)
    elif choice == "Model Training":
        section_model_training(df_processed, model, accuracy, feature_names, X_test, y_test)
    elif choice == "Prediction":
        section_prediction(model, label_encoders)
    elif choice == "About Project":
        st.header("About This Project")
        st.write("""
        - **Project:** Customer Churn Prediction  
        - **Dataset:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
        - **Goal:** Predict if a customer will leave the company (churn) based on their demographic, service, and payment information.
        - **Workflow:** Data loading, preprocessing, EDA, model training, prediction, and Streamlit UI.
        - **Techniques Covered:**
            - Data Preprocessing
            - Feature Encoding
            - Exploratory Data Analysis
            - Classification (Random Forest)
            - Evaluation (accuracy, confusion matrix, feature importance)
            - User Interface (Streamlit)
        - **Developed for:** Data Mining & Machine Learning Project
        """)
        st.info(
    "Please contact the project team; If you have any questions about this project\n"
    "- Wajid Ali: BSDSF22A008\n"
    "- Abdul Manaf: BSDSF22A007\n"
    "- Hammad Farooq: BSDSF22A026\n"
)
    st.markdown("---")
    st.markdown("Made with ❤️ for Data Mining & Machine Learning Project")

if __name__ == "__main__":
    main()
