import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import numpy as np

# Load datasets 
@st.cache_resource
def load_data():
    orders = pd.read_csv('orders.csv')
    order_products_prior = pd.read_csv('order_products__prior.csv')
    order_products_train = pd.read_csv('order_products__train.csv')
    products = pd.read_csv('products.csv')
    
    order_products = pd.concat([order_products_prior, order_products_train], axis=0, ignore_index=True)
    return orders, order_products, products

orders, order_products, products = load_data()

# Preprocess Data
product_counts = order_products.groupby('product_id')['order_id'].count().reset_index().rename(columns={'order_id': 'frequency'})
product_counts = product_counts.sort_values('frequency', ascending=False).head(100).reset_index(drop=True)
product_counts = product_counts.merge(products, on='product_id', how='left')

freq_products = list(product_counts['product_id'])
filtered_order_products = order_products[order_products['product_id'].isin(freq_products)]
filtered_order_products = filtered_order_products.merge(products, on='product_id', how='left')

# Sidebar
st.sidebar.title("Market Basket Analysis")
section = st.sidebar.radio("Navigation", ["Home", "EDA", "Association Rules","Predictive Model"])

# Home
if section == "Home":
    # Add a title and subtitle
    st.markdown("""
    <div style="text-align: center; margin-top: -50px;">
        <h1 style="font-size: 2.5em; color: #4CAF50;">📊 Market Basket Analysis</h1>
        <h3 style="color: #555;">Uncover Customer Insights & Predict Buying Patterns</h3>
    </div>
    """, unsafe_allow_html=True)

    # Add an introduction message
    st.markdown("""
    <div style="margin: 20px 0; font-size: 1.2em; color: #333; text-align: justify;">
        Welcome to the Market Basket Analysis application! This tool is designed to help businesses understand their customers' purchasing behavior. 
        By analyzing historical sales data, you can uncover hidden patterns, optimize product placement, and increase revenue. 
    </div>
    """, unsafe_allow_html=True)

    # Display the features of the application
    st.markdown("""
    <div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px;">
        <h4 style="color: #4CAF50;">🌟 Key Features:</h4>
        <ul style="line-height: 1.8; font-size: 1.1em; color: #333;">
            <li><b>Exploratory Data Analysis (EDA):</b> Analyze product frequency, order trends, and customer habits.</li>
            <li><b>Association Rules Mining:</b> Discover relationships between products using the Apriori algorithm.</li>
            <li><b>Interactive Visualizations:</b> Gain insights with dynamic plots and data tables.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Add a call-to-action
    st.markdown("""
    <div style="text-align: center; margin-top: 30px;">
        <a href="#" style="font-size: 1.2em; color: white; background-color: #4CAF50; padding: 10px 20px; border-radius: 5px; text-decoration: none;">Start Exploring</a>
    </div>
    """, unsafe_allow_html=True)

    # Add a footer with credits
    st.markdown("""
    <div style="text-align: center; font-size: 0.9em; color: #777; margin-top: 50px;">
        Created with ❤️ by [Abhishek singh] 
    </div>
    """, unsafe_allow_html=True)

# EDA
elif section == "EDA":
    st.title("Exploratory Data Analysis")
    
    # Top 10 Products
    st.header("Top 10 Most Frequent Products")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=product_counts.head(10),
        x='frequency', y='product_name', ax=ax, palette='viridis'
    )
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Product Name')
    st.pyplot(fig)
    
    st.write("### Frequency Table of Top 10 Products")
    st.write(product_counts.head(10))
    
    # Order Distribution by Day of Week
    st.header("Order Distribution by Day of Week")
    orders['order_dow'] = orders['order_dow'].replace({
        0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday',
        4: 'Thursday', 5: 'Friday', 6: 'Saturday'
    })
    dow_counts = orders['order_dow'].value_counts().reindex([
        'Sunday', 'Monday', 'Tuesday', 'Wednesday',
        'Thursday', 'Friday', 'Saturday'
    ])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=dow_counts.index, y=dow_counts.values, ax=ax, palette='coolwarm')
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Number of Orders')
    ax.set_title('Orders by Day of Week')
    st.pyplot(fig)
    
    # Order Distribution by Hour of Day
    st.header("Order Distribution by Hour of Day")
    hour_counts = orders['order_hour_of_day'].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=hour_counts.index, y=hour_counts.values, marker='o', ax=ax, color='blue')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Number of Orders')
    ax.set_title('Orders by Hour of Day')
    st.pyplot(fig)
    
    # Reorder Ratio for Top Products
    # Reorder Ratio for Top Products
    st.header("Reorder Ratio for Top 10 Products")
    reorder_ratios = order_products.groupby('product_id')['reordered'].mean().sort_values(ascending=False).head(10)

    # Map product IDs to product names
    top_product_names = products.set_index('product_id').loc[reorder_ratios.index, 'product_name']

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=reorder_ratios.values, y=top_product_names, ax=ax, palette='magma')
    ax.set_xlabel('Reorder Ratio')
    ax.set_ylabel('Product Name')
    ax.set_title('Top 10 Products by Reorder Ratio')
    ax.bar_label(ax.containers[0], fmt='%.2f')  # Add value annotations
    plt.tight_layout()  # Adjust layout to avoid cutting off labels
    st.pyplot(fig)

    
    # Correlation Heatmap
    st.header("Correlation Heatmap")
    correlation_data = orders[['order_dow', 'order_hour_of_day', 'days_since_prior_order']].dropna()
    correlation_data['order_dow'] = correlation_data['order_dow'].astype('category').cat.codes
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_data.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)


# Association Rules
# elif section == "Association Rules":
#     st.title("Association Rules Mining")
    
#     # Transform data into a basket format
#     st.write("Transforming data for Apriori algorithm...")
#     basket = filtered_order_products.groupby(['order_id', 'product_name'])['reordered'].count().unstack().reset_index().fillna(0).set_index('order_id')
    
#     # Encode data for Apriori
#     def encode_units(x):
#         return 1 if x >= 1 else 0
#     basket = basket.applymap(encode_units)
    
#     # Apriori Algorithm
#     st.write("Running Apriori algorithm...")
#     min_support = st.slider("Minimum Support:", min_value=0.001, max_value=0.1, value=0.01, step=0.001)
#     frequent_items = apriori(basket, min_support=min_support, use_colnames=True, low_memory=True)
#     st.write("### Frequent Itemsets")
#     st.dataframe(frequent_items)
    
#     # Association Rules
#     st.write("Generating Association Rules...")
#     metric = st.selectbox("Metric:", ["lift", "confidence", "support"])
#     min_threshold = st.slider(f"Minimum {metric.capitalize()} Threshold:", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
#     rules = association_rules(frequent_items, metric=metric, min_threshold=min_threshold,num_itemsets=3)
    
#     # Filter rules by lift
#     st.write("### Association Rules")
#     if not rules.empty:
#         st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
#         st.write(f"Total Rules: {len(rules)}")
#     else:
#         st.write("No rules found with the selected parameters.")

# Association Rules
elif section == "Association Rules":
    st.title("Association Rules Mining")
    
    # Sampling fraction slider
    st.write("### Data Sampling")
    sample_fraction = st.slider("Select Fraction of Data to Use:", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    sampled_data = filtered_order_products.sample(frac=sample_fraction, random_state=42)
    st.write(f"Using {len(sampled_data)} rows out of {len(filtered_order_products)} total rows.")
    
    # Transform data into a basket format
    st.write("Transforming data for Apriori algorithm...")
    basket = sampled_data.groupby(['order_id', 'product_name'])['reordered'].count().unstack().reset_index().fillna(0).set_index('order_id')
    
    # Encode data for Apriori
    def encode_units(x):
        return 1 if x >= 1 else 0
    basket = basket.applymap(encode_units)
    
    # Apriori Algorithm
    st.write("Running Apriori algorithm...")
    min_support = st.slider("Minimum Support:", min_value=0.001, max_value=0.1, value=0.01, step=0.001)
    frequent_items = apriori(basket, min_support=min_support, use_colnames=True, low_memory=True)
    st.write("### Frequent Itemsets")
    st.dataframe(frequent_items)
    
    # Association Rules
    # st.write("Generating Association Rules...")
    # metric = st.selectbox("Metric:", ["lift", "confidence", "support"])
    # min_threshold = st.slider(f"Minimum {metric.capitalize()} Threshold:", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    # rules = association_rules(frequent_items, metric=metric, min_threshold=min_threshold,num_itemsets=3)
    
    # # Filter rules by lift
    # st.write("### Association Rules")
    # if not rules.empty:
    #     st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
    #     st.write(f"Total Rules: {len(rules)}")
    # else:
    #     st.write("No rules found with the selected parameters.")



# Predictive Model

# Predictive Model Section
elif section == "Predictive Model":
    st.title("Predictive Model Using XGBoost")

    # Load the XGBoost model
    @st.cache_resource
    def load_model():
        with open('xgboost_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model

    model = load_model()

    # Load the dataset
    @st.cache_resource
    def load_final_data():
        return pd.read_pickle('Finaldata.pkl')

    df = load_final_data()

    st.write("### Loaded Dataset for Prediction")
    st.write(df.head())  # Display the first few rows

    # Specify feature columns
    feature_columns = [
        # Replace with the actual list of feature column names
        "total_product_orders_by_user",
        "total_product_reorders_by_user",
        "user_product_reorder_percentage",
        "avg_add_to_cart_by_user",
        "avg_days_since_last_bought",
        "last_ordered_in",
        "is_reorder_3",
        "is_reorder_2",
    ]

    # Ensure feature columns exist in the dataset
    if all(col in df.columns for col in feature_columns):
        # Prepare data for prediction
        D_test = xgb.DMatrix(df[feature_columns])

        # Make predictions
        predictions = model.predict(D_test)
        predictions_binary = [1 if p > 0.5 else 0 for p in predictions]

        # Display predictions
        st.write("### Predictions (Probability):")
        st.write(predictions)
        st.write("### Predictions (Binary):")
        st.write(predictions_binary)

        # Add predictions to the dataset and allow downloading
        predictions_df = df.copy()
        predictions_df['Predicted_Reorder'] = predictions_binary
        st.download_button(
            label="Download Predictions",
            data=predictions_df.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv",
        )
    else:
        st.error("The required feature columns are missing from the dataset. Please check the input data.")
