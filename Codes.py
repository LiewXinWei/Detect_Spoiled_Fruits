import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

# Step 1: Extract Features from image path (reuse your function)
def extract_features_from_image(image_path):
    img = mpimg.imread(image_path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.dtype == np.uint8:
        img = img / 255.0

    mean_r = np.mean(img[:, :, 0])
    mean_g = np.mean(img[:, :, 1])
    mean_b = np.mean(img[:, :, 2])
    std_intensity = np.std(img)
    dark_pixels_ratio = np.sum(np.sum(img, axis=2) < 0.4) / (img.shape[0] * img.shape[1])
    return [mean_r, mean_g, mean_b, std_intensity, dark_pixels_ratio]

# Extract features from PIL Image (for webcam photos)
def extract_features_from_pil(image_pil):
    img = np.array(image_pil.convert('RGB')) / 255.0
    mean_r = np.mean(img[:, :, 0])
    mean_g = np.mean(img[:, :, 1])
    mean_b = np.mean(img[:, :, 2])
    std_intensity = np.std(img)
    dark_pixels_ratio = np.sum(np.sum(img, axis=2) < 0.4) / (img.shape[0] * img.shape[1])
    return [mean_r, mean_g, mean_b, std_intensity, dark_pixels_ratio]

# Step 2: Load Dataset
def load_dataset(base_dir):
    features = []
    labels = []

    for split in ['train', 'valid', 'test']:
        for label_name in ['fresh', 'spoiled']:
            search_path = os.path.join(base_dir, split, label_name, '*', '*.jpg')
            image_paths = glob(search_path)
            for path in image_paths:
                try:
                    feat = extract_features_from_image(path)
                    label = 0 if label_name == 'fresh' else 1
                    features.append(feat)
                    labels.append(label)
                except Exception as e:
                    st.warning(f"❌ Error reading {path}: {e}")
    
    columns = ['mean_R', 'mean_G', 'mean_B', 'std_intensity', 'pct_dark_pixels']
    df = pd.DataFrame(features, columns=columns)
    df['label'] = labels
    return df

# Step 3: Train Model
def train_model(df):
    X = df.drop('label', axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    st.subheader("📊 Classification Report")
    st.text(classification_report(y_test, y_pred))
    st.subheader("🧩 Confusion Matrix")
    st.text(confusion_matrix(y_test, y_pred))

    return model, scaler

# Streamlit App UI
st.title("🍎 Fruit Freshness Classifier with Webcam")

dataset_path = st.text_input("Dataset folder path:", "Fresh and Rotten Fruit.v3i.folder")

if st.button("Load dataset and train model"):
    with st.spinner("Loading dataset and training model..."):
        df = load_dataset(dataset_path)
        if df.empty:
            st.error("⚠️ No image data found. Check your folder path and structure.")
        else:
            st.success(f"✅ Loaded {len(df)} images.")
            st.dataframe(df.head())
            model, scaler = train_model(df)
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler
            st.success("Model training complete. Now you can take a webcam photo below.")

# Webcam input for prediction
if 'model' in st.session_state and 'scaler' in st.session_state:
    img_file_buffer = st.camera_input("Take a photo")

    if img_file_buffer is not None:
        img = Image.open(img_file_buffer)

        features = extract_features_from_pil(img)
        features_scaled = st.session_state['scaler'].transform([features])
        prediction = st.session_state['model'].predict(features_scaled)[0]

        label = "Fresh" if prediction == 0 else "Spoiled"
        st.image(img, caption=f"Prediction: {label}", use_column_width=True)
else:
    st.info("Please load dataset and train the model first.")
