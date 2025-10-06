from pathlib import Path
import csv, joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DATA = Path("data/seed.csv")
OUT = Path("models"); OUT.mkdir(parents=True, exist_ok=True)

def load_data():
    X, y = [], []
    with DATA.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            text  = (row.get("text") or "").strip()
            label = 1 if (row.get("label") or "").strip().lower() == "phishing" else 0
            if text:
                X.append(text); y.append(label)
    if not X:
        raise RuntimeError("No rows in data/seed.csv (needs headers: label,text)")
    return X, y

if __name__ == "__main__":
    X, y = load_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english",
                                  ngram_range=(1,3), min_df=1, max_df=0.95,
                                  max_features=20000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipe.fit(Xtr, ytr)
    acc = pipe.score(Xte, yte)
    print(f"Holdout accuracy: {acc:.3f}")
    print("Vocab size:", len(pipe.named_steps["tfidf"].vocabulary_))

    joblib.dump(pipe, OUT / "pipeline.pkl")
    print("✅ Saved models/pipeline.pkl")
