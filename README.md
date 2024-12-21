# Deepfake Voice Recognition 🎤🔍

Welcome to the **Deepfake Voice Recognition App**, a Streamlit-based web application designed to identify whether an audio file is a deepfake or a real voice. This project leverages machine learning models trained on the **DEEP-VOICE** dataset. The app is a casual exploration of using machine learning to distinguish between real and AI-generated speech.

Try the live app here: [Deepfake Voice Recognition](https://deepfake-voice-recognition-malikfm.streamlit.app)

---

## 🌟 Features

- **Upload Audio**: Users can upload audio files.
- **Choose Model**: Select between two pre-trained models:
  - Random Forest (`rf_model.joblib`)
  - LSTM (`lstm_model.keras`)
- **Deepfake Detection**: The app predicts whether the uploaded voice is real or fake (AI-generated).

---

## 📊 Dataset Overview

This project uses the [DEEP-VOICE dataset](https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition/data), introduced in the study *"Real-time Detection of AI-Generated Speech for DeepFake Voice Conversion"* by Bird and Lotfi (2023). The dataset includes:

- **Real Speech**: Human voices recorded from eight well-known figures.
- **Fake Speech**: AI-generated voices created by converting one speaker's voice to another using Retrieval-based Voice Conversion (RVC).

### Key Dataset Features:
- **Raw Audio**: Available in the `REAL` and `FAKE` directories.
- **Pre-extracted Features**: Stored in `DATASET-balanced.csv`, used for training the models in this project.

### Ethical Considerations:
The dataset was developed to address the rising ethical concerns about generative AI in speech, such as privacy violations and voice misrepresentation. A successful detection system could notify users when AI-generated speech is detected in real-time scenarios like calls or conferences.

---

## 🛠️ Project Structure

```
deepfake-voice-recognition/
├── models/                                 # Trained models
│   ├── rf_model.joblib                     # Random Forest model
│   ├── lstm_model.keras                    # LSTM model
│   └── lstm_scaler.joblib                  # Pre-trained scaler object for feature scaling
├── notebooks/                              # Jupyter notebooks for training and experimentation
│   └── deepfake-voice-recognition.ipynb    # Random Forest and LSTM training notebook
├── app.py                                  # Streamlit app script
├── pyproject.toml                          # Poetry configuration
├── poetry.toml                             # Python dependencies
└── README.md                               # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Poetry (for managing dependencies)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/malikfm/deepfake-voice-recognition.git
   cd deepfake-voice-recognition
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Run the Streamlit app locally:
   ```bash
   poetry run streamlit run app.py
   ```

4. Open the app in your browser: `http://localhost:8501`

---

## 🔍 Model Training

I utilized two distinct approaches for model training, i.e. **Random Forest** and **LSTM**. Detailed training processes and corresponding code can be found in the `notebooks/` folder.

---

## 🔗 Live Demo

Experience the app live: [Deepfake Voice Recognition](https://deepfake-voice-recognition-malikfm.streamlit.app)

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🎉 Acknowledgments
- **Dataset**: [DEEP-VOICE](https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition/data)
- **Study**: *"Real-time Detection of AI-Generated Speech for DeepFake Voice Conversion"* by Bird, J.J. and Lotfi, A. (2023).
