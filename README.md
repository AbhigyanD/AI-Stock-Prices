# 🧠 Advanced AI Stock Price Predictor

An interactive, multi-model stock prediction dashboard built with **JavaScript, HTML, CSS, and TensorFlow.js**. This project leverages deep learning, sentiment analysis, and traditional machine learning models to predict future stock prices.

## 🚀 Features

### 🧠 Advanced ML Models
- **Neural Network (TensorFlow.js)**
  - Multiple hidden layers (64, 32, 16 neurons)
  - Dropout layers for regularization
  - Trained on 60 days of historical data
  - Learns complex, nonlinear patterns
- **Sentiment Analysis**
  - Simulates sentiment scores from news and social media
  - Real-time bullish, bearish, and neutral sentiment indicators
  - Price adjustment based on sentiment scores
- **Logistic Regression**
  - Predicts direction of price movement (up/down)
  - Binary classification using gradient descent
  - Lightweight and interpretable
- **Support Vector Machine**
  - Finds optimal decision boundaries in feature space
  - Handles nonlinear market behavior
  - Margin-based classification
- **Ensemble Model**
  - Combines all models using weighted averaging
  - Improves overall prediction reliability

---

## 📊 Enhanced Features

- ✅ 60-day historical price and volume data
- ✅ Advanced technical indicators:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - EMA (Exponential Moving Average)
- ✅ Real-time model performance metrics:
  - **Accuracy (%)**
  - **MSE (Mean Squared Error)**
- ✅ Confidence scores for each prediction
- ✅ Interactive model toggles (enable/disable models)
- ✅ Multi-timeframe support:
  - **Next Day**
  - **Next Week**
  - **Next Month**
- ✅ Advanced charting using Chart.js with prediction overlays

---

## 📸 Screenshots

> _📈 Prediction cards with confidence scores_  
> _📊 Real-time chart overlay of historical and predicted prices_  
> _📋 Sentiment analysis with dynamic indicators_  
> _⚙️ Toggleable model controls and ensemble forecasting_

---

## 🛠️ Tech Stack

- **Frontend:** HTML, CSS, JavaScript
- **ML Models:** TensorFlow.js, Custom JS implementations
- **Charting:** Chart.js
- **Simulation:** Randomized sentiment & price/volume generation

---

## ⚠️ Disclaimer

> This project is for **educational and demonstrative purposes only**.  
> It does not provide financial advice. Use at your own risk.

---

## 📂 Project Structure
📁 project-root/
├── index.html # Main UI and logic
├── styles.css # Styling
├── script.js # JavaScript ML logic and charting
├── README.md # You're here!



---

## 🚧 Future Improvements

- 🔌 Integrate real-time data via Yahoo Finance or Alpha Vantage APIs
- 🗞️ Use actual news headlines with HuggingFace transformers for sentiment
- 📲 Deploy to Netlify or Vercel with live API backend
- 🧪 Add cross-validation and live retraining support

---

## 👨‍💻 Author

Built with ❤️ by Abhigyan.  
Feel free to contribute, fork, or raise issues!

---

## 📜 License

MIT License
