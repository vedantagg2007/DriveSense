"""DriveSense: cleaned original analysis workflow. See README.md for data setup."""
from features import signal_features, resolve_path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

def load_data(file_path):
    df = pd.read_csv(resolve_path(file_path))
    accel_x = pd.to_numeric(df.iloc[:, -3], errors='coerce').values
    accel_y = pd.to_numeric(df.iloc[:, -2], errors='coerce').values
    accel_z = pd.to_numeric(df.iloc[:, -1], errors='coerce').values
    magnitude_xy = np.sqrt(accel_x ** 2 + accel_y ** 2)
    return (accel_z, magnitude_xy)

def extract_features(z_signal, xy_signal):
    return signal_features(z_signal, extended=False) + signal_features(xy_signal, extended=False)

def main():
    road_files = {'highway': '3 type/highway.csv', 'main road': '3 type/main road.csv', 'neighborhood': '3 type/neighborhood.csv'}
    datasets = {}
    for condition, path in road_files.items():
        try:
            datasets[condition] = load_data(path)
            print(f'Loaded {condition} data with {len(datasets[condition][0])} samples')
        except Exception as e:
            print(f'Error loading {condition}: {str(e)}')
            datasets[condition] = (np.array([]), np.array([]))
    sampling_rate = 836
    segment_length = sampling_rate * 60
    X_train, y_train = ([], [])
    labels = list(datasets.keys())
    for condition_idx, condition in enumerate(labels):
        z_data, xy_data = datasets[condition]
        num_segments = min(len(z_data), len(xy_data)) // segment_length
        for i in range(num_segments):
            z_segment = z_data[i * segment_length:(i + 1) * segment_length]
            xy_segment = xy_data[i * segment_length:(i + 1) * segment_length]
            if not np.isfinite(z_segment).all() or not np.isfinite(xy_segment).all():
                continue
            features = extract_features(z_segment, xy_segment)
            if not np.isnan(features).any():
                X_train.append(features)
                y_train.append(condition_idx)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    if set(y_train.tolist()) != set(range(len(labels))):
        raise ValueError('Need valid 60-second training segments for all three road types. Check the CSV paths and samples.')
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    classifier = SVC(kernel='rbf', C=2.0, gamma='auto', probability=True)
    classifier.fit(X_train_scaled, y_train)
    print('\n--- Training Report ---')
    print(classification_report(y_train, classifier.predict(X_train_scaled), labels=range(len(labels)), target_names=labels, zero_division=0))

    def predict_road_condition(z_data, xy_data):
        segment_length = 836 * 60
        if len(z_data) < segment_length or len(xy_data) < segment_length:
            return {'prediction': 'too short', 'confidence': 0.0}
        z_segment = z_data[:segment_length]
        xy_segment = xy_data[:segment_length]
        try:
            if not np.isfinite(z_segment).all() or not np.isfinite(xy_segment).all():
                return {'prediction': 'invalid segment', 'confidence': 0.0}
            features = extract_features(z_segment, xy_segment)
            features = imputer.transform([features])
            features_scaled = scaler.transform(features)
            prediction = classifier.predict(features_scaled)[0]
            class_index = int(np.where(classifier.classes_ == prediction)[0][0])
            confidence = float(classifier.predict_proba(features_scaled)[0, class_index])
            return {'prediction': labels[prediction], 'confidence': confidence}
        except Exception as e:
            print(f'Prediction error: {str(e)}')
            return {'prediction': 'error', 'confidence': 0.0}
    print('\n--- Sample Predictions ---')
    test_files = {'highway': '3 type/testing/highway.csv', 'main road': '3 type/testing/main road.csv', 'neighborhood': '3 type/testing/neighborhood.csv'}
    all_true = []
    all_pred = []
    for condition, path in test_files.items():
        try:
            z_data, xy_data = load_data(path)
            num_segments = min(len(z_data), len(xy_data)) // segment_length
            print(f'\n{condition.upper()} - {num_segments} segments')
            predictions = []
            confidences = []
            for i in range(num_segments):
                z_seg = z_data[i * segment_length:(i + 1) * segment_length]
                xy_seg = xy_data[i * segment_length:(i + 1) * segment_length]
                result = predict_road_condition(z_seg, xy_seg)
                pred_label = result['prediction']
                if pred_label in labels:
                    all_true.append(labels.index(condition))
                    all_pred.append(labels.index(pred_label))
                    predictions.append(pred_label)
                    confidences.append(result['confidence'])
                    print(f'Segment {i + 1}: {result}')
                else:
                    print(f'Segment {i + 1}: Prediction invalid or too short')
            if predictions:
                overall_pred = max(set(predictions), key=predictions.count)
                avg_conf = np.mean([c for p, c in zip(predictions, confidences) if p == overall_pred])
                print(f'Overall: {overall_pred} (avg confidence: {avg_conf:.2f})')
        except Exception as e:
            print(f'Error testing {condition}: {str(e)}')
    if all_true and all_pred:
        cm = confusion_matrix(all_true, all_pred, labels=range(len(labels)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix on Test Data')
        plt.show()
if __name__ == '__main__':
    main()
