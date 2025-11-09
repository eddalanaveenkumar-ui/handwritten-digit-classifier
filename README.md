# Handwritten Digit Classifier

A complete machine learning project for classifying handwritten digits (0-9) using Random Forest and Neural Networks.

## 🚀 Features

- **Multiple Models**: Random Forest and Neural Network
- **Easy Training**: One-command training for all models
- **Custom Predictions**: Predict digits from your own images
- **Comprehensive Evaluation**: Accuracy reports, confusion matrices, and performance metrics
- **Professional Structure**: Modular, well-organized codebase

## 📁 Project Structure
handwritten-digit-classifier/
├── .gitignore
├── README.md
├── PROJECT_STRUCTURE.md
├── requirements.txt
├── config.yaml
├── fixed_web_app.py
├── cnn_trainer.py
├── cnn_predictor.py
├── improved_trainer.py
├── fine_tune_model.py
├── train.py
├── predict.py
├── simple_web_app.py
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── predictor.py
├── data/
│   └── raw/
│       └── README.md
└── models/
    └── README.md
# Handwritten Digit Classifier

A complete machine learning project for classifying handwritten digits (0-9) with web interface.

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Place MNIST dataset in `data/raw/`
3. Run web app: `python fixed_web_app.py`
4. Open `http://localhost:5000`

## Features

- Multiple models (CNN, Random Forest, Neural Networks)
- Web interface with drawing canvas
- Real-time predictions
- Model fine-tuning