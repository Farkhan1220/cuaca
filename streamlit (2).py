import streamlit as st
import pandas as pd
import numpy as np

# Streamlit App Title
st.title('Linear Regression Weather Prediction')
st.write("This app uses a Linear Regression model to predict weather conditions based on user-provided data.")

# File Upload
uploaded_file = st.file_uploader("klasifikasi_cuaca", type=["csv"])

if uploaded_file is not None:
    # Load Dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Feature Selection
    st.write("### Select Features and Target")
    features = st.multiselect("Select Features for Prediction:", options=df.columns, default=df.columns[:-1])
    target = st.selectbox("Select Target Variable:", options=df.columns, index=len(df.columns)-1)

    if len(features) > 0 and target:
        X = df[features]
        y = df[target]

        # Split Dataset
        test_size = st.slider("Test Size (as a percentage):", 10, 50, 20, step=5) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Standardize Features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train Linear Regression Model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Evaluate Model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write("### Model Performance")
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"R-squared: {r2:.2f}")

        # Plot Actual vs Predicted
        st.write("### Actual vs Predicted")
        fig1, ax1 = plt.subplots()
        ax1.scatter(y_test, y_pred, alpha=0.6)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', color='red')
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Predicted')
        ax1.set_title('Actual vs Predicted')
        st.pyplot(fig1)

        # Residuals Distribution
        st.write("### Residuals Distribution")
        residuals = y_test - y_pred
        fig2, ax2 = plt.subplots()
        sns.histplot(residuals, kde=True, bins=30, ax=ax2)
        ax2.set_title('Residuals Distribution')
        st.pyplot(fig2)
