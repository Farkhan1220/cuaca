import streamlit as st
import pandas as pd
import joblib

# Load pre-trained model (replace 'model.pkl' with the actual model file path)
def load_model():
    return joblib.load('model.pkl')

# Predict weather classification
def predict_weather(model, input_data):
    predictions = model.predict(input_data)
    return predictions

# Streamlit app
def main():
    st.title("Aplikasi Klasifikasi Cuaca")

    # Upload dataset
    uploaded_file = st.file_uploader("klasifikasi_cuaca.csv", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(klasifikas_cuaca)
        st.write("### Pratinjau Data")
        st.write(data.head())

        # Load model
        model = load_model()

        # Ensure the uploaded data has the necessary features
        try:
            st.write("### Prediksi Cuaca")
            predictions = predict_weather(model, data)
            data['Prediction'] = predictions
            st.write(data)

            # Visualization of results
            st.write("### Visualisasi Hasil")
            st.bar_chart(data['Prediction'].value_counts())
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
    else:
        st.info("Silakan unggah file untuk memulai.")

if __name__ == "__main__":
    main()
