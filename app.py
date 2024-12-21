import os
import tempfile

import librosa
import joblib
import numpy as np
import streamlit as st
from sklearn import preprocessing
from tensorflow.keras.models import load_model

# Load models
rf_model = joblib.load("./models/rf_model.joblib")
lstm_model = load_model("./models/lstm_model.keras")


# Function to extract features from audio
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    
    # Compute features
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    rms = np.mean(librosa.feature.rms(y=y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y))
    
    # Compute MFCCs (20 coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs_mean = np.mean(mfccs, axis=1)  # Mean of each coefficient
    
    # Combine all features into a single vector
    features = np.hstack((
        chroma_stft, 
        rms, 
        spectral_centroid, 
        spectral_bandwidth, 
        rolloff, 
        zero_crossing_rate, 
        mfccs_mean
    ))
    
    return features


# Function to predict using the selected model
def predict(model, audio_path):
    if model == "Model 1 (RF)":
        features = extract_features(audio_path)
        print("Random Forest features shape:", features.shape)  # Should be (26,)

        return rf_model.predict([features])[0]
    elif model == "Model 2 (LSTM)":
        features = extract_features(audio_path)

        scaler = joblib.load("./models/lstm_scaler.joblib")
        features = scaler.transform(features.reshape(1, -1))

        # features = np.expand_dims(features, axis=(0, 1))  # Shape: (1, time_steps, 26)
        features = np.expand_dims(features, axis=0).reshape(1, 1, -1)
        print("LSTM features shape:", features.shape)  # Should be (1, time_steps, 26)

        return lstm_model.predict(features)[0][0]


# Streamlit UI
st.title("Deepfake Voice Recognition")

model_choice = st.selectbox("Choose Model", ("Model 1 (RF)", "Model 2 (LSTM)"))

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    result = predict(model_choice, temp_file_path)
    os.remove(temp_file_path)

    print(f"Prediction result: {result}")

    if result > 0.5:
        st.write("The audio is Real")
    else:
        st.write("The audio is Fake")
