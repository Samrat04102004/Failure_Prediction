# 🔧 Failure Prediction Web App

A **Streamlit-based web application** for predicting equipment failure using a machine learning model — optimized and deployed for **real-time use**.

---

## 🚀 Overview

This project provides an **interactive web interface** where users can input equipment parameters and instantly receive predictions on potential failures.

The app leverages a **Random Forest model**, fine-tuned with **Optuna** for superior performance, and is deployed on **Render** for easy access.

---

## ✨ Key Features

- 🔍 **Real-Time Failure Prediction**  
  Enter operational data and get immediate risk assessments.

- 🤖 **Robust Machine Learning**  
  Powered by a **Random Forest model** trained on historical equipment data.

- 🎯 **Hyperparameter Optimization**  
  Utilizes **Optuna** to maximize model accuracy and reliability.

- 🖥️ **User-Friendly Interface**  
  Clean, interactive UI built with **Streamlit** for a seamless experience.

- ☁️ **Scalable Deployment**  
  Hosted on **Render** for reliable and scalable access.

---

## 🌐 Live Demo

👉 [Click here to try the app](https://failure-prediction.onrender.com)

---

## ⚙️ Technologies Used

- **Python 3.13**
- **Streamlit** – Web application framework
- **Pandas** and **NumPy** – Data preprocessing
- **scikit-learn** – Machine learning algorithms
- **Optuna** – Hyperparameter optimization
- **Render** – Cloud deployment

---

## 📁 Project Structure

├──main.py # Streamlit app script

├── requirements.txt # Python dependencies

├── scaler.joblib # Scaler used in model pipeline

└── trained_model.joblib # Trained Random Forest model

---

## 📝 Usage

1. Open the web app in your browser.
2. Input the required equipment or operational parameters.
3. Click **Predict** to view the failure chances and result.
4. Click **View Feature Importance** to see which features had major role on the prediction result.
