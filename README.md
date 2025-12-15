# 🌱 Crop Recommendation System
A machine learning based crop recommendation system that suggests the best crops based on soil nutrients and climate conditions.

>> Features
- Predicts the best crops based on N, P, K, temperature, humidity, pH, and rainfall
- Trains multiple ML models and selects the best performing one automatically
- Top 3 crop suggestions with confidence percentages
- Built with Streamlit for easy web interface
- EDA included for dataset insights

>> Tech Stack
- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

>> How to Run Locally
```bash
# Install dependencies
pip install -r requirements.txt
# Run the Streamlit app
streamlit run streamlit_ui.py