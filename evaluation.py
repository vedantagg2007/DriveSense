"""Keep road quality predictions aligned with original segment identifiers."""


def align_quality_scores(metadata, predictions, scores):
    if len(metadata) != len(predictions):
        raise ValueError("Metadata and prediction counts differ.")
    true_values, predicted_values = [], []
    for (file_name, segment_number, _), prediction in zip(metadata, predictions):
        key = (file_name, segment_number)
        if key in scores:
            true_values.append(float(scores[key]))
            predicted_values.append(float(prediction))
    return true_values, predicted_values
