# Arabic Spoken Digit Recognition System with Detailed Comments

import os  # File and directory operations
import numpy as np  # Numerical operations
import librosa  # Audio processing
import librosa.display  # Displaying audio waveforms
import tensorflow as tf  # Deep learning framework
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score  # Evaluation metrics
from sklearn.svm import SVC  # Support Vector Machine classifier
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns  # Visualization (heatmaps)
from tqdm import tqdm  # Progress bars
from tkinter import Tk, Button, filedialog, Label  # GUI components
import sounddevice as sd  # Audio recording
import soundfile as sf  # Saving and reading audio files

# Configuration paths and labels
DATASET_PATH = "dataset"  # Main dataset folder path
DIGIT_LABELS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']  # Label names
MODEL_FILE = "arabic_digit_model.h5"  # Saved model filename
REC_FILE = "recorded.wav"  # Temporary file for recorded audio

# Function to augment audio using noise or speed changes
def augment_audio(y, sr):
    if np.random.rand() < 0.5:
        noise = np.random.randn(len(y))  # Generate noise
        y_aug = y + 0.005 * noise  # Add noise to signal
    else:
        speed = np.random.uniform(0.9, 1.1)  # Random speed change
        y_aug = librosa.resample(y, orig_sr=sr, target_sr=int(sr * speed))  # Change speed by resampling
    return y_aug

# Function to extract features (Mel spectrograms) from audio
def extract_features(file_path, max_len=128, augment=False):
    y, sr = librosa.load(file_path, sr=16000)  # Load audio file
    if augment:
        y = augment_audio(y, sr)  # Apply augmentation if specified
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)  # Extract mel spectrogram
    mel_db = librosa.power_to_db(mel, ref=np.max)  # Convert to decibel scale
    # Normalize feature length
    if mel_db.shape[1] < max_len:
        pad_width = max_len - mel_db.shape[1]
        mel_db = np.pad(mel_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :max_len]
    return mel_db

# Function to load the entire dataset and apply augmentation
def load_dataset():
    X, y = [], []
    print("Loading and augmenting dataset...")
    for label_idx, label in enumerate(DIGIT_LABELS):
        folder = os.path.join(DATASET_PATH, label)  # Get folder for label
        for file in tqdm(os.listdir(folder), desc=f"Processing {label}"):
            if file.endswith(".wav"):
                file_path = os.path.join(folder, file)
                X.append(extract_features(file_path))  # Original
                y.append(label_idx)
                X.append(extract_features(file_path, augment=True))  # Augmented
                y.append(label_idx)
    return np.array(X), np.array(y)

# Build deep learning model: CNN + BiLSTM
def build_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Reshape((*input_shape, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Reshape((-1, 64)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train deep learning model and compare with SVM
def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    model = build_model(X.shape[1:], len(DIGIT_LABELS))

    print("\n🔹 Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    model.save(MODEL_FILE)
    print(f"\n✅ Model saved as {MODEL_FILE}")

    y_pred = np.argmax(model.predict(X_test), axis=1)  # Predict using DL model

    # Show classification metrics
    print("\n📝 Classification Report (CNN + RNN):")
    print(classification_report(y_test, y_pred, target_names=DIGIT_LABELS))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=DIGIT_LABELS, yticklabels=DIGIT_LABELS)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix - CNN + RNN")
    plt.show()

    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.tight_layout()
    plt.show()

    # Train and evaluate SVM
    print("\n⚡ Training SVM for comparison...")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    svm = SVC(kernel='linear')
    svm.fit(X_train_flat, y_train)
    y_svm_pred = svm.predict(X_test_flat)

    print("\n📝 Classification Report (SVM):")
    print(classification_report(y_test, y_svm_pred, target_names=DIGIT_LABELS))

    # Confusion matrix for SVM
    cm_svm = confusion_matrix(y_test, y_svm_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Purples', xticklabels=DIGIT_LABELS, yticklabels=DIGIT_LABELS)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix - SVM")
    plt.show()

    return model

# Predict digit from a WAV file
def predict_file(model, file_path):
    features = extract_features(file_path)
    features = np.expand_dims(features, axis=0)
    pred_idx = np.argmax(model.predict(features), axis=1)[0]
    digit = DIGIT_LABELS[pred_idx]
    print(f"🎯 Predicted Digit: {digit}")
    return digit

# Record audio from microphone
def record_audio(duration=2, fs=16000):
    print("🎤 Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write(REC_FILE, recording, fs)
    print(f"✅ Audio saved to {REC_FILE}")
    return REC_FILE

# Play audio file
def play_audio(file_path):
    data, fs = sf.read(file_path, dtype='float32')
    sd.play(data, fs)
    sd.wait()

# Build GUI application
def build_gui(model):
    def choose_file():
        label_result.config(text="Processing...")
        root.update_idletasks()
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if file_path:
            play_audio(file_path)
            digit = predict_file(model, file_path)
            label_result.config(text=f"Predicted: {digit}")

    def record_and_predict():
        label_result.config(text="Recording...")
        root.update_idletasks()
        file_path = record_audio()
        play_audio(file_path)

        y, sr = librosa.load(file_path, sr=16000)
        plt.figure(figsize=(8, 2))
        librosa.display.waveshow(y, sr=sr)
        plt.title("Recorded Audio")
        plt.show()

        digit = predict_file(model, file_path)
        label_result.config(text=f"Predicted (mic): {digit}")

    root = Tk()
    root.title("Arabic Digit Recognizer")
    Button(root, text="Choose WAV File", command=choose_file, font=("Arial", 14)).pack(pady=10)
    Button(root, text="Record Live (Mic)", command=record_and_predict, font=("Arial", 14)).pack(pady=10)
    global label_result
    label_result = Label(root, text="", font=("Arial", 16))
    label_result.pack(pady=20)
    root.mainloop()

# Main routine
if __name__ == "__main__":
    print("🚀 Loading dataset...")
    X, y = load_dataset()
    print(f"✅ Loaded {X.shape[0]} samples (with augmentation)")

    if os.path.exists(MODEL_FILE):
        print(f"🔹 Loading model from {MODEL_FILE}")
        model = tf.keras.models.load_model(MODEL_FILE)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else:
        model = train_and_evaluate(X, y)

    build_gui(model)
