import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load your data
# -----------------------------
swiggy_cleaned = pd.read_csv("D:\GUVI DS 2025\Swiggy Recommendation\swiggy_cleaned_mdl.csv")   # original file
swiggy_encoded = pd.read_csv("D:\GUVI DS 2025\Swiggy Recommendation\swiggy_encoded_mdl")   # numeric encoded file

# -----------------------------
# Compute cosine similarity
# -----------------------------
cosine_sim = cosine_similarity(swiggy_encoded)

# -----------------------------
# Helper: get similar restaurants by index
# -----------------------------
def get_similar_restaurants(index, top_n=5):
    sim_scores = list(enumerate(cosine_sim[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:top_n+1]]
    return swiggy_cleaned.iloc[top_indices][['name','address','city','cuisine','cost','rating']]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("< Swiggy Restaurant Finder")

# Convert rating and cost to numeric
swiggy_cleaned['rating'] = pd.to_numeric(swiggy_cleaned['rating'], errors='coerce')
swiggy_cleaned['cost']   = pd.to_numeric(swiggy_cleaned['cost'], errors='coerce')

# User input filters
city_input = st.text_input("Enter City")
cuisine_input = st.text_input("Enter Cuisine")
rating_input = st.number_input("Minimum Rating", min_value=0.0, max_value=5.0, step=0.1)
cost_input = st.number_input("Maximum Cost", min_value=0, step=50)

if st.button("Find Restaurants"):
    # Apply filters
    filtered = swiggy_cleaned.copy()
    if city_input:
        filtered = filtered[filtered['city'].str.contains(city_input, case=False, na=False)]
    if cuisine_input:
        filtered = filtered[filtered['cuisine'].str.contains(cuisine_input, case=False, na=False)]
    if rating_input > 0:
        filtered = filtered[filtered['rating'] >= rating_input]
    if cost_input > 0:
        filtered = filtered[filtered['cost'] <= cost_input]

    if not filtered.empty:
        st.write("### Matching Restaurants")
        st.dataframe(filtered[['name','address','city','cuisine','cost','rating']])
    else:
        st.warning("� Your exact preference is not available.")
        st.write("### Showing Similar Restaurants Instead")
        
        # Pick a random restaurant as reference (or use closest by rating/cost)
        ref_index = 0  # you can improve this by choosing based on user inputs
        recommendations = get_similar_restaurants(ref_index, top_n=5)
        st.dataframe(recommendations)