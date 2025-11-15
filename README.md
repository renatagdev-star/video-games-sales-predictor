# 🎮 Video Game Sales Prediction

This project predicts whether a video game will achieve **GOOD** or **BAD** global sales using machine learning and engineered statistical features.

---

## 🧠 Project Overview

- **Exploratory Data Analysis (EDA)**  
  Statistical exploration, visualization, distribution analysis, and outlier detection.

- **Data Preprocessing**  
  Cleaning the dataset, handling missing values, removing invalid entries, and preparing the data for modeling.

- **AutoGluon Modeling**  
  Automated model benchmarking to identify the strongest algorithm.

- **Feature Engineering**  
  Creation of statistical, interaction, ranking, and aggregated features to boost predictive performance.

- **Model Evaluation**  
  LightGBM achieved the best F1-test score of **0.74**.

- **Manual Retraining**  
  The selected LightGBM model was retrained separately to reduce memory usage and improve performance.

- **Deployment**  
  The final model is integrated into a Streamlit app enabling real-time predictions.

---

## ⚙️ Technologies Used

- Python, Pandas, NumPy, Matplotlib, Seaborn  
- AutoGluon, LightGBM, Scikit-learn  
- Streamlit  

---

## 🌍 Live Demo

Try the hosted application:

**https://renatagdev-star-video-games-sales-predictor-app-fu783i.streamlit.app/**

---
## 🌍 App Preview:

https://github.com/renatagdev-star/video-games-sales-predictor/blob/main/app_preview.JPG

## 📈 Results

- Accuracy: **73%**  
- Macro F1-score: **0.73**  
- Best Model: **LightGBM (F1-test = 0.74)**

---

## 📦 Model File

Trained model: `lightgbm_sales_classifier.pkl`
