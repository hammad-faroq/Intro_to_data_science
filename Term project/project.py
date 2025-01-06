import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# Load dataset
@st.cache_data
def load_data():
    file_path = 'happiness.csv'  # Ensure this is the correct path in your app
    data = pd.read_csv(file_path)
    return data


data = load_data()

# Shared variables for features and target
numeric_features = data.select_dtypes(include=np.number).columns.tolist()
categorical_features = data.select_dtypes(include='object').columns.tolist()

# Initialize session state variables
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None

# Preprocessing
st.title("World Happiness Report Analysis")

st.sidebar.header("Project Sections")
section = st.sidebar.selectbox(
    "Navigate to:",
    ("Introduction", "Exploratory Data Analysis", "Data Preprocessing", "Machine Learning Model", "Conclusion")
)

if section == "Introduction":
    st.header("Introduction")
    st.write(
        "This project analyzes the World Happiness Report dataset. The goal is to explore the data, preprocess it, and apply a machine learning model to gain insights.")
    st.write(
        "The dataset contains happiness scores and related factors for various countries, providing an opportunity to explore the relationships between these factors.")
    st.write("### Dataset Overview")
    st.dataframe(data)
    st.write("Shape of the dataset:", data.shape)

elif section == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")

    st.write("### Summary Statistics")
    st.write(data.describe())

    st.write("### Missing Values")
    st.write(data.isnull().sum())

    st.write("### Correlation Matrix (Numeric Data Only)")
    numeric_data = data.select_dtypes(include=np.number)
    corr = numeric_data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.write("### Feature Distributions")
    for column in numeric_data.columns:
        fig, ax = plt.subplots()
        sns.histplot(data[column], kde=True, ax=ax)
        ax.set_title(f"Distribution of {column}")
        st.pyplot(fig)

elif section == "Data Preprocessing":
    st.header("Data Preprocessing")

    st.write("### Handling Missing Values")
    # Handle missing values for numeric features by imputing the mean
    data[numeric_features] = data[numeric_features].fillna(data[numeric_features].mean())
    st.write("Missing values in numeric columns have been imputed with column means.")

    st.write("### Encoding Categorical Variables")
    if 'Country or region' in categorical_features:
        data = pd.get_dummies(data, columns=['Country or region'], drop_first=True)
        st.write("Categorical variables have been encoded.")

    st.write("### Scaling Numerical Features")
    scaler = StandardScaler()
    data[numeric_features] = scaler.fit_transform(data[numeric_features])
    st.write("Numeric features have been scaled.")

    st.write("### Splitting Data")
    if 'Score' in data.columns:
        X = data.drop(columns=['Score'])  # Assuming 'Score' is the target column
        y = data['Score']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Store the split data in session state
        st.session_state.X_train = X_train
        st.session_state.y_train = y_train
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test

        st.write("Data has been split into training and testing sets.")
    else:
        st.error("'Score' column not found in dataset!")

    # Add a button to confirm preprocessing
    if st.button('Complete Data Preprocessing'):
        st.success("Data preprocessing completed successfully!")

elif section == "Machine Learning Model":
    st.header("Machine Learning Model")

    # Check if data preprocessing is completed
    if st.session_state.X_train is None or st.session_state.y_train is None:
        st.error("Please complete the 'Data Preprocessing' section before training the model.")
    else:
        st.write("### Training a Random Forest Regressor")
        model = RandomForestRegressor(random_state=42)
        model.fit(st.session_state.X_train, st.session_state.y_train)

        st.write("### Evaluating the Model")
        y_pred = model.predict(st.session_state.X_test)
        mse = round(mean_squared_error(st.session_state.y_test, y_pred),3)
        r2 = round(r2_score(st.session_state.y_test, y_pred),3)

        st.write(f"Mean Squared Error: {mse}")
        st.write(f"R-squared: {r2}")

        # st.write("### Feature Importances")
        # feature_importances = pd.DataFrame({
        #     'Feature': st.session_state.X_train.columns,
        #     'Importance': model.feature_importances_
        # }).sort_values(by='Importance', ascending=False)
        # st.dataframe(feature_importances)
        #
        # fig, ax = plt.subplots()
        # sns.barplot(x='Importance', y='Feature', data=feature_importances, ax=ax)
        # ax.set_title("Feature Importances")
        # st.pyplot(fig)
        st.write("### Feature Importances")
        feature_importances = pd.DataFrame({
            'Feature': st.session_state.X_train.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        # Select the top 10 features
        top_10_features = feature_importances.head(7)

        # Display the top 10 feature importances
        st.dataframe(top_10_features)

        # Plot the top 10 features
        fig, ax = plt.subplots()
        sns.barplot(x='Importance', y='Feature', data=top_10_features, ax=ax)
        ax.set_title("Top 10 Feature Importances")
        st.pyplot(fig)


elif section == "Conclusion":
    st.header("Conclusion")
    st.write("The project provided insights into the factors influencing happiness scores across countries.")
    st.write(
        "The Random Forest Regressor performed well in predicting happiness scores, with an R-squared value indicating the model's accuracy.")
    st.write("Future work could explore other machine learning models and deeper feature engineering.")

st.sidebar.markdown("---")
st.sidebar.write("Developed with Streamlit")
