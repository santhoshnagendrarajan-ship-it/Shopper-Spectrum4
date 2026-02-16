# Retail Analytics Dashboard

A comprehensive retail analytics application that performs customer segmentation and product recommendations using RFM (Recency, Frequency, Monetary) analysis and machine learning clustering.

## Project Overview

This project provides an end-to-end solution for analyzing online retail data, including:

- **Data Cleaning & Preprocessing**: Remove duplicates, missing values, outliers, and cancelled transactions
- **RFM Analysis**: Segment customers based on purchasing behavior
- **Customer Clustering**: Group customers into segments using K-Means clustering
- **Product Recommendations**: Provide personalized product recommendations using collaborative filtering
- **Interactive Dashboard**: Streamlit-based web interface for visualization and predictions

## Project Structure

```
├── app.py                 # Streamlit web application
├── analysis.ipynb         # RFM analysis notebook
├── or.ipynb              # Exploratory data analysis notebook
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Features

### 1. Data Preprocessing
- Remove duplicate transactions
- Handle missing customer IDs
- Exclude cancelled invoices
- Remove negative/zero quantities and prices
- Detect and remove outliers using IQR method

### 2. RFM Analysis
- **Recency**: Days since last purchase
- **Frequency**: Number of transactions per customer
- **Monetary**: Total amount spent by customer

### 3. Customer Segmentation
- K-Means clustering (default: 4 clusters)
- Cluster profiles analysis
- Customer segment prediction based on RFM values

### 4. Product Recommendations
- Collaborative filtering using cosine similarity
- Recommendation engine based on product purchase patterns
- Returns top 5 similar products for any given product

### 5. Interactive Dashboard
Three main pages:
- **Home**: Welcome and feature overview
- **Recommendation**: Search for similar products
- **Clustering**: Predict customer segment using RFM values

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. Clone or download this project
2. Navigate to the project directory:


3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure you have the required data file:
   - `online_retail.csv` - Online retail transaction dataset

5. Ensure trained models exist:
   - `kmeans_rfm_model.pkl` - Pre-trained KMeans model
   - `rfm_scaler.pkl` - Fitted StandardScaler for RFM values

## Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Running the Analysis Notebooks

1. **analysis.ipynb**: RFM analysis and clustering
   - Data cleaning and preprocessing
   - RFM metric calculation
   - K-Means clustering with elbow method
   - Cluster profile analysis

2. **or.ipynb**: Exploratory Data Analysis
   - Transaction volume by country
   - Top-selling products
   - Purchase trends over time
   - Transaction value distribution
   - RFM analysis and clustering

## Dataset

The project uses the Online Retail dataset with the following columns:
- `InvoiceNo`: Transaction identifier
- `StockCode`: Product code
- `Description`: Product name
- `Quantity`: Number of items purchased
- `InvoiceDate`: Transaction date and time
- `UnitPrice`: Price per unit
- `CustomerID`: Customer identifier
- `Country`: Customer country

## Models & Techniques

- **K-Means Clustering**: Customer segmentation based on RFM metrics
- **StandardScaler**: Feature normalization for clustering
- **Cosine Similarity**: Product recommendation similarity calculation
- **Elbow Method**: Optimal cluster determination

## Customer Segments

Default clustering maps customers to 4 segments:
- **High-Value**: Regular, high-spending customers
- **Regular**: Consistent customers with moderate spending
- **Occasional**: Infrequent purchasers
- **At-Risk**: Inactive or low-value customers

## Dependencies

See `requirements.txt` for all dependencies and their versions.

Core packages:
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms
- **joblib**: Model serialization
- **numpy**: Numerical computations
- **matplotlib**: Data visualization

## How to Use Each Component

### Product Recommendation
1. Navigate to the "Recommendation" page
2. Enter a product name from your dataset
3. Click "Get Recommendations"
4. View 5 similar products based on collaborative filtering

### Customer Segmentation
1. Navigate to the "Clustering" page
2. Enter RFM values:
   - **Recency**: Days since last purchase (0-1000)
   - **Frequency**: Number of purchases (0-500)
   - **Monetary**: Total spending amount (0-50000)
3. Click "Predict Cluster"
4. View predicted customer segment

## Performance Considerations

- Data preprocessing removes approximately 30-50% of outliers
- K-Means clustering trains on scaled RFM metrics for better performance
- Product similarity matrix computed using cosine distance
- Application loads pre-trained models for fast predictions

## Future Enhancements

- Add RFM score calculation and ranking
- Implement additional clustering algorithms (DBSCAN, Hierarchical)
- Add time-series forecasting for sales predictions
- Implement A/B testing framework for recommendations
- Add export functionality for customer segments
- Include more recommendation algorithms

## Troubleshooting

### Issue: "Product not found"
- Verify product name matches exactly with dataset
- Check data preprocessing hasn't removed all instances

### Issue: Model loading errors
- Ensure `kmeans_rfm_model.pkl` and `rfm_scaler.pkl` exist
- Rerun analysis notebooks to regenerate models

### Issue: Data file not found
- Verify `online_retail.csv` is in the project root directory
- Check file path in code

## License

This project is for educational and analytical purposes.

## Author

GUVI Project 4 - Retail Analytics Analysis

## Contact & Support

For questions or issues, review the notebook documentation and inline comments.

---

**Last Updated**: February 2026
