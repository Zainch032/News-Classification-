from flask import Flask, render_template, request
import pickle
import numpy as np
import os
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import string
from nltk.stem import SnowballStemmer
from collections import Counter
import pandas as pd
import seaborn as sns
import matplotlib

# Use non-interactive backend for server environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)

# Initialize preprocessing resources
punctuation = set(string.punctuation)
stop = set(ENGLISH_STOP_WORDS)
stemmer = SnowballStemmer("english")

# Get absolute paths for PythonAnywhere
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "Linear_Svc.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "model", "tfidf.pkl")
DATA_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
STATIC_DIR = os.path.join(BASE_DIR, "static")
CHART_DIR = os.path.join(STATIC_DIR, "charts")

# Load model + TF-IDF
try:
    model = pickle.load(open(MODEL_PATH, "rb"))
    tfidf = pickle.load(open(TFIDF_PATH, "rb"))
    print(f"✅ Model loaded from: {MODEL_PATH}")
    print(f"✅ TF-IDF loaded from: {TFIDF_PATH}")
    print(f"Model type: {type(model)}")
except Exception as e:
    print(f"❌ Error loading files: {e}")
    model = None 
    tfidf = None

def preprocess(value):
    value = value.lower().split()
    value = [w for w in value if w.isalnum()]
    value = [w for w in value if w not in stop and w not in punctuation]
    value = [stemmer.stem(w) for w in value]
    return " ".join(value)

def map_prediction(pred):
    mapping = {1: "World", 2: "Sports", 3: "Business", 4: "Technology"}
    return mapping.get(pred[0], "Unknown")

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    sum_e_x = e_x.sum(axis=0)
    return e_x / sum_e_x if sum_e_x != 0 else np.zeros_like(e_x)

def get_probabilities(model, vector):
    """Get probabilities from model (works with SVC and LinearSVC)."""
    try:
        # Try predict_proba first (for SVC with probability=True)
        return model.predict_proba(vector)[0]
    except AttributeError:
        try:
            # For LinearSVC (no predict_proba)
            decision = model.decision_function(vector)
            
            # Handle different decision_function return shapes
            if len(decision.shape) == 2:
                scores = decision[0]  # Multi-class, one-vs-rest
            else:
                scores = decision  # Binary or weird shape
            
            # Apply softmax
            return softmax(scores)
        except Exception as e:
            print(f"Probability extraction error: {e}")
            return None


def load_dataset():
    """Load train/test CSVs; return combined DataFrame or None on failure."""
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        print("⚠️ Dataset not found. Skipping visualization build.")
        return None
    try:
        df_train = pd.read_csv(TRAIN_PATH, on_bad_lines="skip")
        df_test = pd.read_csv(TEST_PATH)
        return pd.concat([df_train, df_test], ignore_index=True)
    except Exception as exc:
        print(f"⚠️ Failed to load dataset: {exc}")
        return None


def ensure_chart_dir():
    try:
        os.makedirs(CHART_DIR, exist_ok=True)
    except Exception as exc:
        print(f"⚠️ Could not create chart directory: {exc}")


def save_class_distribution_chart(df):
    values = df["Class Index"].value_counts().sort_index()
    labels_map = {1: "World", 2: "Sports", 3: "Business", 4: "Technology"}
    labels = [labels_map.get(idx, str(idx)) for idx in values.index]
    explode = [0.05, 0.01, 0.1, 0.03]

    def autopct_format(pct, allvals):
        absolute = int(round(pct / 100 * sum(allvals)))
        return f"{pct:.2f}%\n({absolute})"

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(
        values,
        labels=labels,
        explode=explode[: len(values)],
        autopct=lambda pct: autopct_format(pct, values),
        startangle=90,
        textprops={"fontsize": 9},
    )
    ax.set_title("Class Distribution of News Dataset")
    fig.tight_layout()
    output_path = os.path.join(CHART_DIR, "class_distribution.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return os.path.relpath(output_path, STATIC_DIR).replace(os.sep, "/")


def save_top_words_chart(df, class_index, filename):
    subset = df[df["Class Index"] == class_index]
    if subset.empty:
        return None

    processed = subset["Description"].astype(str).apply(preprocess)
    words = " ".join(processed).split()
    most_common = Counter(words).most_common(15)
    if not most_common:
        return None

    df_common = pd.DataFrame(most_common, columns=["Word", "Count"])
    label_map = {1: "World", 2: "Sports", 3: "Business", 4: "Technology"}
    label = label_map.get(class_index, f"Class {class_index}")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=df_common, x="Word", y="Count", palette="viridis", ax=ax)
    ax.set_title(f"Top Words: {label}")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    output_path = os.path.join(CHART_DIR, filename)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return os.path.relpath(output_path, STATIC_DIR).replace(os.sep, "/")


def build_visualizations():
    """Generate chart images from the dataset at startup."""
    ensure_chart_dir()
    df = load_dataset()
    if df is None:
        return {}

    charts = {}
    try:
        charts["class_distribution"] = save_class_distribution_chart(df)
        charts["business_top_words"] = save_top_words_chart(df, 3, "business_top_words.png")
        charts["technology_top_words"] = save_top_words_chart(df, 4, "technology_top_words.png")
    except Exception as exc:
        print(f"⚠️ Failed to build visualizations: {exc}")

    # Remove entries that failed to generate
    return {k: v for k, v in charts.items() if v}


CHART_FILES = build_visualizations()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    input_text = ""
    
    if request.method == "POST":
        input_text = request.form.get("input_text", "")
        
        if input_text.strip():
            if model is None:
                result = "Error: Model not loaded. Check server logs."
            elif tfidf is None:
                result = "Error: TF-IDF vectorizer not loaded."
            else:
                try:
                    # Preprocess and transform
                    processed = preprocess(input_text)
                    transformed = tfidf.transform([processed])
                    
                    # Predict
                    prediction = model.predict(transformed)
                    result = map_prediction(prediction)
                    
                    # Get confidence scores
                    probs = get_probabilities(model, transformed)
                    if probs is not None and len(probs) >= 4:
                        confidence = {
                            "World": float(probs[0]) * 100,
                            "Sports": float(probs[1]) * 100,
                            "Business": float(probs[2]) * 100,
                            "Technology": float(probs[3]) * 100,
                        }
                except Exception as e:
                    result = f"Prediction error: {str(e)}"
                    print(f"❌ Prediction error: {e}")
    
    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        input_text=input_text,
    )


@app.route("/dashboard", methods=["GET"])
def dashboard():
    return render_template("dashboard.html", charts=CHART_FILES)


if __name__ == "__main__":
    app.run(debug=True)