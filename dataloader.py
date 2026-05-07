
import numpy as np


DIGIT_ROWS = 28
DIGIT_COLS = 28
FACE_ROWS  = 70
FACE_COLS  = 60


def _parse_images(filepath: str, rows: int, cols: int) -> np.ndarray:

    with open(filepath, "r") as f:
        lines = f.readlines()

    lines = [line.rstrip("\n") for line in lines]

    total_lines = len(lines)
    n_images = total_lines // rows

    features = np.zeros((n_images, rows * cols), dtype=np.float32)

    for img_idx in range(n_images):
        for row_idx in range(rows):
            line = lines[img_idx * rows + row_idx]
            line = line.ljust(cols)
            for col_idx in range(cols):
                if line[col_idx] != " ":
                    features[img_idx, row_idx * cols + col_idx] = 1.0

    return features


def _parse_labels(filepath: str) -> np.ndarray:

    with open(filepath, "r") as f:
        labels = [int(line.strip()) for line in f if line.strip()]
    return np.array(labels, dtype=np.int32)


def load_digit_data(images_path: str, labels_path: str):

    X = _parse_images(images_path, DIGIT_ROWS, DIGIT_COLS)
    y = _parse_labels(labels_path)
    assert len(X) == len(y), (
        f"Mismatch: {len(X)} images but {len(y)} labels in {images_path}"
    )
    return X, y


def load_face_data(images_path: str, labels_path: str):

    X = _parse_images(images_path, FACE_ROWS, FACE_COLS)
    y = _parse_labels(labels_path)
    assert len(X) == len(y), (
        f"Mismatch: {len(X)} images but {len(y)} labels in {images_path}"
    )
    return X, y


def get_subset(X: np.ndarray, y: np.ndarray, fraction: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(y)
    k = max(1, int(round(n * fraction)))
    indices = rng.choice(n, size=k, replace=False)
    return X[indices], y[indices]



if __name__ == "__main__":
    import os

    DATA_DIR = "." 

    def p(name, X, y):
        print(f"{name:30s}  X={X.shape}  y={y.shape}  "
              f"labels={sorted(set(y.tolist()))}")

    for split, img, lbl in [
        ("digit train",      "trainingimages",   "traininglabels"),
        ("digit validation", "validationimages", "validationlabels"),
        ("digit test",       "testimages",       "testlabels"),
    ]:
        X, y = load_digit_data(
            os.path.join(DATA_DIR, img),
            os.path.join(DATA_DIR, lbl),
        )
        p(split, X, y)

    for split, img, lbl in [
        ("face train",      "facedatatrain",       "facedatatrainlabels"),
        ("face validation", "facedatavalidation",  "facedatavalidationlabels"),
        ("face test",       "facedatatest",        "facedatatestlabels"),
    ]:
        X, y = load_face_data(
            os.path.join(DATA_DIR, img),
            os.path.join(DATA_DIR, lbl),
        )
        p(split, X, y)

    X_train, y_train = load_digit_data(
        os.path.join(DATA_DIR, "trainingimages"),
        os.path.join(DATA_DIR, "traininglabels"),
    )
    X_sub, y_sub = get_subset(X_train, y_train, 0.1)
    print(f"\n10% digit training subset: {X_sub.shape}")