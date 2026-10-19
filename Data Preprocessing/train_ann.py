import os
import zipfile
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

ZIP_PATH = r"Data Preprocessing\Data Set.zip"
STAGING_DIR = r"Backend\staging"

def main():
    print("--- Phase 2: Reproducible ANN Training ---")
    print(f"Loading dataset from {ZIP_PATH}...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        csv_filename = [f for f in z.namelist() if f.endswith('.csv')][0]
        with z.open(csv_filename) as f:
            df = pd.read_csv(f)
            
    df.columns = df.columns.str.strip()
    
    # EXACT MAPPING BASED ON YOUR CSV
    text_col = 'Resume'
    label_col = 'Role'
    
    # 1. Clean the data
    df = df.dropna(subset=[text_col, label_col])
    df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    
    # 2. Filter out categories with only 1 instance to avoid train_test_split crash
    class_counts = df[label_col].value_counts()
    valid_classes = class_counts[class_counts > 1].index
    df = df[df[label_col].isin(valid_classes)].reset_index(drop=True)
    
    # 3. Encode labels
    le = LabelEncoder()
    df['Encoded_Label'] = le.fit_transform(df[label_col])
    num_classes = len(le.classes_)
    
    print(f"\nDiscovered {num_classes} unique categories (Roles):")
    print(le.classes_)
    
    # 4. Split data safely
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col], df['Encoded_Label'], 
        test_size=0.2, random_state=42, stratify=df['Encoded_Label']
    )
    
    # 5. Fit TF-IDF Vectorizer
    print("\nFitting TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    X_train_vec = tfidf.fit_transform(X_train).toarray()
    X_test_vec = tfidf.transform(X_test).toarray()
    
    # 6. Build the Keras ANN Model
    print(f"\nBuilding model with input_dim=5000 and output_dim={num_classes}...")
    model = Sequential([
        Dense(128, activation='relu', input_dim=5000),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # 7. Train
    print("\nTraining the model...")
    model.fit(X_train_vec, y_train, validation_data=(X_test_vec, y_test), epochs=10, batch_size=32)
    
    # 8. Evaluate
    print("\n--- Evaluation Report ---")
    y_pred_probs = model.predict(X_test_vec)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Macro F1: {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Weighted F1: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    
    # 9. Save Artifacts Safely to Staging
    print(f"\nSaving artifacts to {STAGING_DIR}...")
    os.makedirs(STAGING_DIR, exist_ok=True)
    
    model.save(os.path.join(STAGING_DIR, "resume_ann_model.keras"))
    with open(os.path.join(STAGING_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
    with open(os.path.join(STAGING_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
        
    print("Training complete! Artifacts safely stored in staging.")

if __name__ == "__main__":
    main()
