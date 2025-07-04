# 🌱 AI-Powered Crop Recommender

An AI-based system that suggests the best crop to grow based on soil and environmental conditions like NPK, pH, rainfall, temperature, and humidity.

## 🚀 Features
- Predicts the optimal crop using machine learning
- Easy-to-use interface
- Built with Python and scikit-learn

## ⚙️ Setup
1. Clone the repo:
```bash
git clone https://github.com/codrdecodr/AI-Powered-Crop-Recommender.git
cd AI-Powered-Crop-Recommender
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python app.py
```

*Or use `streamlit run app.py` if you're using Streamlit.*

## 📌 Example

Input:

```
N: 90, P: 42, K: 43, Temp: 25.6°C, Humidity: 80%, pH: 6.5, Rainfall: 200mm
```

Output:

```
Recommended Crop: Rice
```
