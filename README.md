# 📊 **Market Basket Analysis Application**

An interactive web application for Market Basket Analysis (MBA) designed to uncover purchasing patterns, generate association rules, and build predictive models for better decision-making.

---

## 🚀 **Features**

### 🏠 Home Section
- Provides an overview of the application.
- User-friendly interface with styled components for better appeal.

### 🔍 Exploratory Data Analysis (EDA)
- **Top Products Analysis**: Visualizes the top 10 most frequently purchased products.
- **Order Trends**:
  - Orders by the day of the week.
  - Orders by the hour of the day.
- **Reorder Ratio Analysis**: Highlights products with the highest reorder ratios.
- **Correlation Heatmap**: Displays relationships between variables like:
  - Day of the week.
  - Order hour of the day.
  - Days since the last order.

### 🔗 Association Rules Mining
- Utilizes the Apriori algorithm to extract frequent itemsets.
- Parameters include:
  - Support threshold (adjustable via a slider).
  - Sampling for faster processing.
- Generates insights on product combinations for cross-selling opportunities. *(Note: Full rule generation implementation planned.)*

### 🤖 Predictive Modeling
- Implements an **XGBoost** model for predictions.
- Uses preprocessed features stored in a pickled dataset.
- Displays dataset preview for transparency.

---

## 🔧 **Technologies Used**
- **Data Handling**: Pandas
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: XGBoost, Sklearn
- **UI Development**: Streamlit
- **Association Rules Mining**: mlxtend

---

## 💡 **Highlights**
- **Dynamic Controls**: Sliders and dropdowns to adjust key parameters.
- **Optimized Performance**: Sampling mechanisms and caching for faster execution.
- **Visual Appeal**: Clean, interactive charts and tables for intuitive insights.

---

## 📈 **Future Enhancements**
- Add a **detailed association rules table** with metrics like confidence and lift.
- Improve **error handling** for user-friendly debugging.
- Include **real-time predictions** for the Predictive Model section.
- Expand visualization options for deeper insights.

---

## ⚙️ **How to Run**
1. Clone the repository:
   ```bash
   git clone <repository-url>
