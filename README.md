# Swiggy-Restaurant-Recommendation System

## 📌 Project Overview
This project builds a recommendation engine for Swiggy users.  
It suggests restaurants or dishes based on user preferences, past orders, and contextual features such as cuisine type, ratings, and location.

## 🚀 Features
- Collaborative filtering for personalized recommendations
- Content-based filtering using cuisine, price, and ratings
- Hybrid recommendation combining multiple approaches
- Interactive Streamlit app for testing recommendations

## 🛠️ Tech Stack
- **Python** (pandas, numpy, scikit-learn)
- **Streamlit** (for UI)
- **MLflow** (for experiment tracking)
- **Joblib/Pickle** (for saving models)

## 📂 Project Structure

swiggy-recommendation/ │── data/          
# Datasets (orders, restaurants, ratings) │── notebooks/    
# Jupyter notebooks for exploration │── models/               
# Saved recommendation models │── app.py              
# Streamlit app │── train.py             
# Training script │── README.md       
# Project documentation
