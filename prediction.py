"""DriveSense: cleaned original analysis workflow. See README.md for data setup."""
from features import signal_features, resolve_path
from evaluation import align_quality_scores
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os
import matplotlib.pyplot as plt
SAMPLING_RATE = 836
SEGMENT_LENGTH = 60 * SAMPLING_RATE

def extract_features(signal_data):
    return signal_features(signal_data, extended=True)

def load_dataset(file_paths, score_lists):
    X_all, y_all = ([], [])
    for file_path, manual_scores in zip(file_paths, score_lists):
        df = pd.read_csv(resolve_path(file_path), low_memory=False)
        accel_z = pd.to_numeric(df.iloc[:, -1], errors='coerce').values
        num_segments = len(accel_z) // SEGMENT_LENGTH
        for i in range(min(num_segments, len(manual_scores))):
            start = i * SEGMENT_LENGTH
            end = (i + 1) * SEGMENT_LENGTH
            segment_z = accel_z[start:end]
            if not np.isfinite(segment_z).all():
                continue
            features_z = extract_features(segment_z)
            X_all.append(features_z)
            y_all.append(manual_scores[i])
    return (np.array(X_all), np.array(y_all))

def load_unlabeled_dataset_with_labels(file_paths):
    X_all = []
    meta_info = []
    for file_path in file_paths:
        df = pd.read_csv(resolve_path(file_path), low_memory=False)
        road_labels = df.iloc[:, 0].values
        accel_z = pd.to_numeric(df.iloc[:, -1], errors='coerce').values
        num_segments = len(accel_z) // SEGMENT_LENGTH
        for i in range(num_segments):
            start = i * SEGMENT_LENGTH
            end = (i + 1) * SEGMENT_LENGTH
            segment_z = accel_z[start:end]
            segment_labels = road_labels[start:end]
            if not np.isfinite(segment_z).all():
                continue
            features_z = extract_features(segment_z)
            X_all.append(features_z)
            label = most_common_label(segment_labels)
            meta_info.append((os.path.basename(file_path), i + 1, label))
    return (np.array(X_all), meta_info)

def most_common_label(labels):
    labels = labels[~pd.isnull(labels)]
    if len(labels) == 0:
        return 'Unknown'
    return pd.Series(labels).mode().iloc[0]

def main():
    train_files = ['3 type/highway.csv', '3 type/main road.csv', '3 type/neighborhood.csv']
    test_files = ['3 type/testing/highway.csv', '3 type/testing/main road.csv', '3 type/testing/neighborhood.csv']
    manual_scores_train = [[8, 9, 9, 9, 10, 9, 9, 9, 10, 8, 10, 10, 10, 9, 8, 9, 9, 4, 3, 4, 3, 3, 5, 5, 5, 6, 4, 3, 8, 9, 9, 9, 8, 3, 4, 4, 4, 4, 4, 3, 4, 5, 5, 4, 4, 3, 8, 7, 7, 8, 7, 8, 7, 7, 8, 8, 9, 8, 8, 7, 7, 7, 7, 8, 7, 8, 6, 7, 8, 7, 7, 8, 7, 8, 8, 7, 7], [8, 6, 7, 7, 8, 9, 9, 7, 7, 7, 7, 8, 8, 9, 9, 9, 7, 8, 8, 8, 8, 5, 7, 7, 6, 7, 8, 8, 8, 8, 7, 6, 7, 5, 4, 4, 3, 6, 7, 4, 3, 6, 4, 3, 5, 7, 4, 9, 8, 8, 7, 8, 7, 7, 9, 9, 6, 6, 7, 7, 8, 7, 7, 8, 8, 7], [6, 7, 7, 7, 7, 6, 5, 6, 6, 7, 7, 7, 7, 7, 5, 6, 6, 7, 8, 8, 8, 8, 9, 1, 4, 3, 4, 2, 1, 2, 5, 6, 7, 7, 2, 2, 6, 2, 3, 7, 6, 7, 8, 7, 8, 8, 7, 7, 7, 6, 5, 8, 7, 8, 8, 7, 6, 8, 8, 7, 8, 8, 8, 7]]
    X_train, y_train = load_dataset(train_files, manual_scores_train)
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    X_test, meta_info = load_unlabeled_dataset_with_labels(test_files)
    X_test = imputer.transform(X_test)
    X_test = scaler.transform(X_test)
    y_pred = model.predict(X_test)
    print('\nPredicted Road Quality Scores on Test Data:')
    print('File\t\t\tSegment\tLabel\t\tScore')
    print('-' * 60)
    for (file_name, segment_num, label), pred in zip(meta_info, y_pred):
        pred_rounded = int(round(pred))
        print(f'{file_name:<24} Segment {segment_num:<3} {label:<12} Score = {pred_rounded}')
    real_all = [7, 6, 4, 5, 5, 5, 4, 4, 4, 4, 5, 5, 6, 5, 4, 4, 6, 7, 7, 3, 5, 4, 3, 4, 4, 5, 4, 5, 7, 6] + [6, 6, 6, 6, 5, 6, 5, 7, 7] + [7, 5, 6, 4, 4, 5, 7]
    test_segment_counts = {'highway.csv': 30, 'main road.csv': 9, 'neighborhood.csv': 7}
    score_map = {}
    score_offset = 0
    for file_name, count in test_segment_counts.items():
        for segment_num in range(1, count + 1):
            score_map[file_name, segment_num] = real_all[score_offset + segment_num - 1]
        score_offset += count
    aligned_true, predicted_all = align_quality_scores(meta_info, y_pred, score_map)
    real_all = aligned_true
    if len(real_all) < 2:
        raise ValueError('At least two aligned test segments are needed for R².')
    r2 = r2_score(real_all, predicted_all)
    print(f'\nOverall R² Score (continuous predictions): {r2:.4f}')
    sorted_indices = np.argsort(real_all)
    real_sorted = np.array(real_all)[sorted_indices]
    pred_sorted = np.array(predicted_all)[sorted_indices]
    plt.figure(figsize=(12, 6))
    plt.plot(real_sorted, label='Real Scores', marker='o', color='blue')
    plt.plot(pred_sorted, label='Predicted Scores', marker='x', color='red')
    plt.title(f'Real vs Predicted Road Quality Scores (R² = {r2:.2f})')
    plt.xlabel('Segments sorted by observed score')
    plt.ylabel('Road Quality Score')
    plt.ylim(0, 10)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    main()
