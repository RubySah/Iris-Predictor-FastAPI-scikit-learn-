# Train an iris classifier on the UCI Iris dataset and save it as model/iris_model.pkl.

# Data source: https://archive.ics.uci.edu/dataset/53/iris

import io
import pickle
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "iris.data"
MODEL_PATH = BASE_DIR / "model" / "iris_model.pkl"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/53/iris.zip"

# iris.data has no header row, so we name the columns ourselves.
COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]


def download_data():
    print(f"Downloading {UCI_ZIP_URL} ...")
    with urllib.request.urlopen(UCI_ZIP_URL) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    DATA_PATH.parent.mkdir(exist_ok=True)
    DATA_PATH.write_bytes(archive.read("iris.data"))
    print(f"Saved {DATA_PATH}")


def main():
    if not DATA_PATH.exists():
        download_data()

    # skip_blank_lines handles the empty line at the end of the file
    df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS, skip_blank_lines=True).dropna()
    print(f"Loaded {len(df)} rows. Classes: {sorted(df['species'].unique())}")

    X = df[COLUMNS[:4]].values
    y = df["species"].values  # e.g. "Iris-setosa"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random forests don't need feature scaling
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(f"Test accuracy: {accuracy_score(y_test, model.predict(X_test)):.3f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "target_names": list(model.classes_)}, f)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
