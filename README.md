# Fraud Detection

🚨 A web-based machine learning project to detect fraudulent transactions using Python and Flask.

## 📌 Project Overview

This project is a part of a final assignment / undergraduate thesis (Skripsi) focused on implementing machine learning for detecting fraud patterns. It uses a trained model (`fraud.pkl`) and provides a web interface using Flask.

## 🛠️ Features

- Upload or input transaction data
- Predict whether a transaction is fraudulent or not
- Simple and responsive web interface

## 🧠 Machine Learning

- Algorithm: Gradient Boosting
- Input Features: amount, location, transactionType, Timestamp, isFraud
- Trained model saved as `fraud.pkl`

## Model Evaluation - Gradient Boosting

![Gradient Boosting Evaluation](img/gradient boosting.png)

## 🗂️ Project Structure
```bash
fraud-detection/
│
├── templates/ # HTML templates for the web interface
├── app.py # Main Flask application
├── fraud.pkl # Serialized ML model
├── requirements.txt # Python dependencies
├── Procfile # For deployment (e.g., Heroku)
├── .gitignore # Files and folders to be ignored by Git
└── README.md # Project documentation
```
## ⚙️ Installation

# Clone the repository
```bash
git clone https://github.com/agungdhrs/fraud-detection.git
cd fraud-detection
```
# Install dependencies
```bash
pip install -r requirements.txt
```
# Run the app
```bash
python app.py
```
