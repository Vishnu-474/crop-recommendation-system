import streamlit as st
import numpy as np
import pandas as pd
import joblib
#Loading_Model_and_scaler
model = joblib.load("models/crop_model.pkl")
scaler = joblib.load("models/scaler.pkl")
#Crop_label_dictionary
crop_dict = {
    1: '🌾 Rice', 2: '🌽 Maize', 3: '🧵 Jute', 4: '👕 Cotton', 5: '🥥 Coconut',
    6: '🍈 Papaya', 7: '🍊 Orange', 8: '🍎 Apple', 9: '🍈 Muskmelon', 10: '🍉 Watermelon',
    11: '🍇 Grapes', 12: '🥭 Mango', 13: '🍌 Banana', 14: '🍎 Pomegranate',
    15: '🥣 Lentil', 16: '🫘 Blackgram', 17: '🌱 Mungbean', 18: '🌿 Mothbeans',
    19: '🌾 Pigeonpeas', 20: '🫘 Kidneybeans', 21: '🌰 Chickpea', 22: '☕ Coffee'
}
#Information_of_Project
st.sidebar.title("📊 Project Info")
st.sidebar.markdown("""
**Crop Recommendation System**  
Using soil nutrients, weather, and ML to suggest the best crops 🌱

- Built with Streamlit  
- Trained on 22 crops  
- Shows top 3 suggestions  
""")
st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 Developed by Vishnu")

st.title("🌿 Smart Crop Recommendation")
st.markdown("Give your field's stats and get the top 3 crop suggestions.")

#Sliders
st.header("📥 Input Conditions")
N = st.slider("Nitrogen (N)", 0, 140, 80)
P = st.slider("Phosphorous (P)", 0, 140, 40)
K = st.slider("Potassium (K)", 0, 200, 50)
temperature = st.slider("Temperature (°C)", 10, 45, 25)
humidity = st.slider("Humidity (%)", 10, 100, 80)
ph = st.slider("Soil pH", 3.0, 10.0, 6.5)
rainfall = st.slider("Rainfall (mm)", 0, 300, 100)

#Predictions
if st.button("🚀 Predict Crops"):
    features = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                            columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    scaled_features = scaler.transform(features)
    probs = model.predict_proba(scaled_features)[0]

    top_indices = np.argsort(probs)[::-1][:3]
    predicted_labels = model.classes_

    st.subheader("🌱 Top 3 Crop Suggestions")
    for i, idx in enumerate(top_indices, start=1):
        label = predicted_labels[idx]
        crop = crop_dict[label]
        confidence = probs[idx] * 100
        st.success(f"{i}. {crop} — **{confidence:.2f}% confidence**")
        st.progress(int(confidence))
#Credits
st.markdown("---")
st.markdown("© 2025 Vishnu | Built with Passion and ❤️ using Streamlit")
