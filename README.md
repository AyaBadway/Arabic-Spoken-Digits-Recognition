# 🗣 Arabic Digit Speech Recognition

An end-to-end speech recognition app for recognizing **Arabic spoken digits (0-9)** using deep learning and machine learning models. The app supports **live microphone recording** and **WAV file uploads** to predict spoken digits, along with a user-friendly GUI interface.

---

## 🚀 About the App

This project:

- Recognizes **Arabic spoken digits** (`zero` to `nine`) from audio files or live microphone input.
- Uses **deep learning (CNN + Bidirectional LSTM)** and **SVM models** for training and comparison.
- Includes **data augmentation techniques** (noise injection, speed variation) to enhance accuracy.
- Provides a **Tkinter GUI** for easy interaction.

---

## 🧠 Models Used

### 1️⃣ CNN + Bidirectional LSTM

- **Convolutional layers (Conv2D)** for spatial feature extraction from Mel spectrograms.
- **MaxPooling** to reduce spatial dimensionality.
- **BatchNormalization** to stabilize and speed up training.
- **Bidirectional LSTM** to capture temporal dependencies in audio.
- **Dense + Dropout** for final classification.

### 2️⃣ Support Vector Machine (SVM)

- Linear kernel SVM trained on **flattened spectrogram features**.
- Used as a baseline for comparison against deep learning model.

---

## 🗂 Dataset

Orginal dataset source:
- **Dataset**: [Arabic Speech Commands Dataset](https://github.com/abdulkaderghandoura/arabic-speech-commands-dataset/tree/v1.0/dataset)
- **Source**: Zenodo / GitHub
- **Classes**: 10 (zero to nine)
- **Samples**: ~300 `.wav` recordings per digit, spoken by native Arabic speakers.
- **Data Augmentation**: Noise addition + speed variation → ~6000 total samples.

Note: dataset folder contains the only needed data that used on training this model other extra data from the orginal source removed.
---

## 💻 Code Functionality

| Part | Function |
|-------|----------|
| `augment_audio` | Applies random noise or speed variation to audio |
| `extract_features` | Generates Mel spectrogram features |
| `load_dataset` | Loads and augments the dataset |
| `build_model` | Builds the CNN + LSTM architecture |
| `train_and_evaluate` | Trains the model and evaluates on test set, shows reports and confusion matrix |
| `predict_file` | Predicts digit from a given audio file |
| `record_audio` | Records live audio from mic |
| `play_audio` | Plays back audio file |
| `build_gui` | GUI for uploading file / live recording and displaying prediction |

---

## 📈 Example Metrics

| Model | Accuracy | Precision | Recall |
|--------|----------|-----------|--------|
| CNN + BiLSTM | 0.94 | 0.94 | 0.94 |
| SVM | 0.81 | 0.81 | 0.81 |

---

## ⚙ Installation

### 1️⃣ Prerequisites

- Python 3.11
- `pip`

Then install the following ⚙ Requirements: 

TensorFlow 2.x

librosa

matplotlib

seaborn

scikit-learn

tqdm

soundfile

sounddevice

pyaudio (via pipwin)


=> One command installation: pip install numpy librosa tensorflow scikit-learn matplotlib seaborn tqdm sounddevice soundfile

### 2️⃣ Clone the repository and run the app.


=> git clone https://github.com/AyaBadway/Arabic-Spoken-Digits-Recognition

=> cd arabic-digit-speech-recognition

=> then run the app file using this command: python arabic_digit_recognizer.py

=> App will work and you can start use the app and test.



