"""DriveSense: cleaned original analysis workflow. See README.md for data setup."""
from features import signal_features, resolve_path
from condition_model import fit_condition_model
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
import matplotlib.pyplot as plt

def load_data(file_path):
    df = pd.read_csv(resolve_path(file_path), low_memory=False)
    condition_labels = df.iloc[:, 0].values
    accel_x = pd.to_numeric(df.iloc[:, -3], errors='coerce').values
    accel_y = pd.to_numeric(df.iloc[:, -2], errors='coerce').values
    accel_z = pd.to_numeric(df.iloc[:, -1], errors='coerce').values
    magnitude_2d = np.sqrt(accel_x ** 2 + accel_y ** 2)
    return (condition_labels, magnitude_2d, accel_z)

def extract_features(signal_data):
    return signal_features(signal_data, extended=True)

def main():
    sampling_rate = 836
    segment_length = sampling_rate * 60
    condition_labels = ['smooth', 'grooves', 'bumpy']
    road_files = {'highway': 'combined/highway.csv', 'main road': 'combined/main road.csv', 'neighborhood': 'combined/neighborhood.csv'}
    X_train, y_train = ([], [])
    for file_path in road_files.values():
        labels, magnitude_2d, accel_z = load_data(file_path)
        num_segments = len(magnitude_2d) // segment_length
        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = (i + 1) * segment_length
            segment_2d = magnitude_2d[start_idx:end_idx]
            segment_z = accel_z[start_idx:end_idx]
            segment_labels = labels[start_idx:end_idx]
            valid_labels = pd.Series(segment_labels).dropna()
            if valid_labels.empty:
                continue
            majority_label = valid_labels.mode().iloc[0]
            if majority_label in condition_labels:
                if not np.isfinite(segment_2d).all() or not np.isfinite(segment_z).all():
                    continue
                features_2d = extract_features(segment_2d)
                features_z = extract_features(segment_z)
                features = features_2d + features_z
                if not np.isnan(features).any():
                    X_train.append(features)
                    y_train.append(majority_label)
    print('Original segment counts:', Counter(y_train))
    label_to_int = {label: idx for idx, label in enumerate(condition_labels)}
    int_to_label = {v: k for k, v in label_to_int.items()}
    y_train_int = np.array([label_to_int[label] for label in y_train])
    X_train = np.asarray(X_train, dtype=float)
    classifier, X_val, y_val, best_k, best_score = fit_condition_model(X_train, y_train_int)
    print(f'Best k: {best_k}; training-fold CV accuracy: {best_score:.3f}')
    y_val_pred = classifier.predict(X_val)
    print('\n--- Classification Report ---')
    print(classification_report(y_val, y_val_pred, labels=range(len(condition_labels)), target_names=condition_labels, zero_division=0))
    cm = confusion_matrix(y_val, y_val_pred, labels=range(len(condition_labels)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=condition_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix on Validation Set')
    plt.show()
    print('\n--- Predictions ---')
    for i in range(len(X_val)):
        features = X_val[i].reshape(1, -1)
        proba = classifier.predict_proba(features)[0]
        pred_idx = np.argmax(proba)
        confidence = proba[pred_idx]
        prediction = condition_labels[int(classifier.classes_[pred_idx])]
        true_label = int_to_label[y_val[i]]
        print(f'Sample {i + 1}: Label = {true_label}, Predicted = {prediction}, Confidence = {confidence:.2f}')
if __name__ == '__main__':
    main()
