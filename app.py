import streamlit as st
import pandas as pd
import joblib
import pickle
import urllib.request
import os

# ---------------- CONFIG ----------------
FEATURE_MAPS_PATH = "feature_maps.pkl"
MODEL_PATH = "lightgbm_sales_classifier.pkl"

# Google Drive direct download link
MODEL_URL = "https://drive.google.com/uc?export=download&id=1pKT2LAU-fimnEopM0QgG0e7QJaCgb7o-"


# ---------------- DOWNLOAD MODEL ----------------
def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Google Drive...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    # safety check
    if os.path.getsize(MODEL_PATH) < 50000:
        raise RuntimeError("Model file is corrupted or incomplete. Check sharing settings.")


# ---------------- LOAD ARTIFACTS ----------------
@st.cache_resource
def load_artifacts():
    download_model_if_needed()

    feature_maps = joblib.load(FEATURE_MAPS_PATH)

    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    features = artifact["features"]
    label_mapping = artifact["label_mapping"]
    best_threshold = artifact["best_threshold"]

    return feature_maps, model, features, label_mapping, best_threshold


feature_maps, model, features, label_mapping, best_threshold = load_artifacts()
inv_label_mapping = {v: k for k, v in label_mapping.items()}


# ---------------- FEATURE ENGINEERING ----------------
def prepare_features(platform, genre, publisher, feature_maps, features):
    row = {
        'Platform': platform,
        'Genre': genre,
        'Publisher': publisher,
        'Publisher_avg_sales': feature_maps['publisher_avg_sales_map'].get(publisher, 0),
        'Genre_avg_sales': feature_maps['genre_avg_sales_map'].get(genre, 0),
        'Platform_avg_sales': feature_maps['platform_avg_sales_map'].get(platform, 0),
        'Platform_Genre': f"{platform}_{genre}",
        'Platform_Publisher': f"{platform}_{publisher}",
        'Genre_Publisher': f"{genre}_{publisher}",
        'Publisher_rank': feature_maps['publisher_rank_map'].get(publisher, 0),
        'Genre_rank': feature_maps['genre_rank_map'].get(genre, 0),
        'Platform_rank': feature_maps['platform_rank_map'].get(platform, 0),
    }

    df = pd.DataFrame([row])

    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category')

    return df[features]


# ---------------- UI ----------------
st.title("🎮 Video Game Sales Quality Prediction")
st.write("Predict whether a new game will sell **GOOD** or **BAD** based on Platform, Genre and Publisher.")

platform_options = sorted(feature_maps['platform_rank_map'].keys())
genre_options = sorted(feature_maps['genre_rank_map'].keys())
publisher_options = sorted(feature_maps['publisher_rank_map'].keys())

col1, col2, col3 = st.columns(3)

with col1:
    platform = st.selectbox("Platform", platform_options)

with col2:
    genre = st.selectbox("Genre", genre_options)

with col3:
    publisher = st.selectbox("Publisher", publisher_options)


if st.button("Predict"):
    test_df = prepare_features(
        platform=platform,
        genre=genre,
        publisher=publisher,
        feature_maps=feature_maps,
        features=features
    )

    prob_good = model.predict_proba(test_df)[:, 1][0]
    pred_label_num = int(prob_good >= best_threshold)
    pred_name = inv_label_mapping[pred_label_num].upper()

    st.subheader("Result")
    st.write(f"Prediction: **{pred_name}**")
    st.write(f"Probability GOOD: `{prob_good:.3f}`")
    st.write(f"Threshold: `{best_threshold:.2f}`")

    st.caption("Model: LightGBM classifier trained on historical video game sales.")


with st.expander("Show model input row"):
    try:
        st.dataframe(test_df)
    except:
        st.write("Click Predict first.")
