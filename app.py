import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Load models and data
kmeans = joblib.load('kmeans_rfm_model.pkl')
scaler = joblib.load('rfm_scaler.pkl')
# Load only the Description column to avoid building a large dense pivot table
data = pd.read_csv('online_retail.csv', usecols=['Description'])

# Derive product names from unique descriptions (memory-efficient)
product_names = data['Description'].dropna().unique().tolist()
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(product_names)

product_similarity = cosine_similarity(tfidf_matrix,tfidf_matrix)

def recommend_products(product_name, top_n=5):
    if product_name not in product_names:
        return []
    idx = product_names.index(product_name)
    sim_scores = list(enumerate(product_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    recommended = [product_names[i] for i, score in sim_scores[1:top_n+1]]
    return recommended

def predict_cluster(recency, frequency, monetary):
    X = scaler.transform([[recency, frequency, monetary]])
    cluster = kmeans.predict(X)[0]
    # Map cluster to label (adjust as per your cluster_profiles logic)
    cluster_map = {0: 'High-Value', 1: 'Regular', 2: 'Occasional', 3: 'At-Risk'}
    return cluster_map.get(cluster, f'Cluster {cluster}')

# Streamlit App
st.set_page_config(page_title="Retail Analytics App", layout="wide")

# Sidebar navigation
page = st.sidebar.selectbox("Choose Page", ["Home", "Recommendation", "Clustering"])

if page == "Home":
    st.title("🏠 Retail Analytics Dashboard")
    st.markdown("""
    Welcome to the Retail Analytics App!  
    - **Recommendation:** Find similar products using collaborative filtering.  
    - **Clustering:** Predict customer segment using RFM values.
    """)

elif page == "Recommendation":
    st.title("🛒 Product Recommendation")
    st.markdown("Enter a product name to get 5 similar products based on purchase history.")
    product_input = st.text_input("Product Name")
    if st.button("Get Recommendations"):
        recs = recommend_products(product_input)
        if recs:
            st.success("Recommended Products:")
            for prod in recs:
                st.markdown(f"- **{prod}**")
        else:
            st.error("Product not found. Please check the name and try again.")

elif page == "Clustering":
    st.title("🔍 Customer Segmentation")
    st.markdown("Input RFM values to predict the customer segment.")
    recency = st.number_input("Recency (Days since last purchase)", min_value=0, value=30)
    frequency = st.number_input("Frequency (Number of transactions per customer)", min_value=0, value=5)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=100.0)
    if st.button("Predict Cluster"):
        segment = predict_cluster(recency, frequency, monetary)
        st.success(f"Predicted Segment: **{segment}**")