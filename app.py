import streamlit as st
import pandas as pd
import joblib
import pickle

# ---------------- CONFIG ----------------
FEATURE_MAPS_PATH = "feature_maps.pkl"  
MODEL_PATH = "lightgbm_sales_classifier.pkl" 

@st.cache_resource
def load_artifacts():
    # 1) load feature maps
    feature_maps = joblib.load(FEATURE_MAPS_PATH)

    # 2) load artifact dict iz pickle-a
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    features = artifact["features"]
    label_mapping = artifact["label_mapping"]
    best_threshold = artifact["best_threshold"]

    return feature_maps, model, features, label_mapping, best_threshold


# Load everything
feature_maps, model, features, label_mapping, best_threshold = load_artifacts()

# Reverse label mapping
inv_label_mapping = {v: k for k, v in label_mapping.items()}


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

    # convert object cols to categorical
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category')

    # reorder columns to match training
    df = df[features]

    return df


# ---------------- UI ----------------
st.title("🎮 Video Game Sales Quality Prediction")
st.write("Predict whether a new game will sell **GOOD** or **BAD** based on Platform, Genre and Publisher.")


platform_options = list(feature_maps['platform_rank_map'].keys())
genre_options = list(feature_maps['genre_rank_map'].keys())
publisher_options = list(feature_maps['publisher_rank_map'].keys())

col1, col2, col3 = st.columns(3)

with col1:
    platform = st.selectbox("Platform", sorted(platform_options))

with col2:
    genre = st.selectbox("Genre", sorted(genre_options))

with col3:
    publisher = st.selectbox("Publisher", sorted(publisher_options))


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
    st.write(f"Decision Threshold: `{best_threshold:.2f}`")

    st.caption("Model: LightGBM classifier trained on historical video game sales data.")


with st.expander("Show model input row"):
    try:
        st.dataframe(test_df)
    except:
        st.write("Click Predict first.")

