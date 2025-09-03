# Credit Card Fraud Detection Analytics

A comprehensive machine learning project for detecting fraudulent credit card transactions using advanced analytics and classification algorithms.

## 📊 Project Overview

This project analyzes credit card transaction data to build predictive models that can identify fraudulent transactions in real-time. Using a dataset from Kaggle, we implement various machine learning techniques to tackle the challenges of fraud detection in highly imbalanced datasets.

## 🎯 Objectives

- Analyze patterns in fraudulent vs legitimate credit card transactions
- Build robust machine learning models for fraud detection
- Handle class imbalance effectively using various sampling techniques
- Evaluate model performance using appropriate metrics for fraud detection
- Provide actionable insights for fraud prevention strategies

## 📁 Dataset Information

**Source:** Kaggle Credit Card Fraud Detection Dataset

**Key Characteristics:**
- **Size:** 284,807 transactions
- **Features:** 31 (28 PCA-transformed features + Time, Amount, Class)
- **Target Variable:** Class (0 = Legitimate, 1 = Fraudulent)
- **Class Distribution:** Highly imbalanced (~0.17% fraudulent transactions)
- **Time Period:** 2 days of transactions

**Features Description:**
- `V1-V28`: PCA-transformed features (anonymized for privacy)
- `Time`: Seconds elapsed between transaction and first transaction
- `Amount`: Transaction amount
- `Class`: Target variable (0/1)

## 🛠️ Technologies Used

- **Python 3.8+**
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine Learning:** Scikit-learn, XGBoost, LightGBM
- **Imbalanced Learning:** Imbalanced-learn (SMOTE, ADASYN)
- **Model Evaluation:** Classification metrics, ROC-AUC analysis

## 📋 Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
imbalanced-learn>=0.8.0
jupyter>=1.0.0
```

## 🚀 Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2. **Create virtual environment:**
```bash
python -m venv fraud_detection_env
source fraud_detection_env/bin/activate  # On Windows: fraud_detection_env\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download dataset:**
- Download the dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Place `creditcard.csv` in the `data/` directory

## 📊 Project Structure

```
credit-card-fraud-detection/
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv
│   └── processed/
│       └── preprocessed_data.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── utils.py
│
├── models/
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   └── xgboost_model.pkl
│
├── results/
│   ├── model_performance.csv
│   └── confusion_matrices/
│
├── visualizations/
│   ├── feature_distributions.png
│   ├── correlation_heatmap.png
│   └── roc_curves.png
│
├── requirements.txt
├── README.md
└── main.py
```

## 🔍 Analysis Pipeline

### 1. Exploratory Data Analysis (EDA)
- Distribution analysis of features
- Class imbalance investigation
- Correlation analysis
- Temporal patterns exploration
- Amount distribution analysis

### 2. Data Preprocessing
- Missing value handling
- Feature scaling (StandardScaler, RobustScaler)
- Outlier detection and treatment
- Feature selection techniques

### 3. Handling Class Imbalance
- **Undersampling:** Random undersampling
- **Oversampling:** SMOTE, ADASYN
- **Ensemble Methods:** Balanced Random Forest
- **Cost-sensitive Learning:** Class weight adjustment

### 4. Model Development
- **Baseline Models:** Logistic Regression, Decision Trees
- **Ensemble Methods:** Random Forest, Gradient Boosting
- **Advanced Models:** XGBoost, LightGBM
- **Deep Learning:** Neural Networks (optional)

### 5. Model Evaluation
- **Metrics:** Precision, Recall, F1-Score, ROC-AUC, PR-AUC
- **Cross-validation:** Stratified K-Fold
- **Confusion Matrix Analysis**
- **Feature Importance Analysis**

## 📈 Key Results

### Model Performance Summary

| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|-------|-----------|--------|----------|---------|--------|
| Logistic Regression | 0.85 | 0.78 | 0.81 | 0.92 | 0.74 |
| Random Forest | 0.89 | 0.82 | 0.85 | 0.95 | 0.81 |
| XGBoost | 0.91 | 0.85 | 0.88 | 0.97 | 0.86 |

### Key Insights
- Transaction amount and timing patterns are strong fraud indicators
- PCA features V14, V12, and V10 show highest importance
- Ensemble methods significantly outperform baseline models
- SMOTE oversampling improves recall while maintaining precision

## 🏃‍♂️ Usage

### Quick Start
```python
# Load and run the complete pipeline
python main.py
```

### Custom Analysis
```python
from src.model_training import FraudDetectionModel
from src.data_preprocessing import DataPreprocessor

# Initialize preprocessor and model
preprocessor = DataPreprocessor()
model = FraudDetectionModel()

# Load and preprocess data
data = preprocessor.load_data('data/raw/creditcard.csv')
X_train, X_test, y_train, y_test = preprocessor.prepare_data(data)

# Train model
model.train(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

## 📊 Evaluation Metrics

For fraud detection, we prioritize:
- **Recall (Sensitivity):** Minimize false negatives (missed fraud)
- **Precision:** Minimize false positives (legitimate transactions flagged)
- **PR-AUC:** Better than ROC-AUC for imbalanced datasets
- **Cost Analysis:** Consider business impact of different error types

## 🚧 Challenges & Solutions

### Class Imbalance (0.17% fraud)
- **Solution:** SMOTE oversampling + ensemble methods
- **Result:** Improved recall from 45% to 85%

### Feature Interpretability
- **Challenge:** PCA-transformed features
- **Solution:** SHAP values for model explainability

### Real-time Performance
- **Requirement:** Low latency for transaction processing
- **Solution:** Model optimization and feature selection

## 🔮 Future Enhancements

- **Real-time Streaming:** Apache Kafka integration
- **Deep Learning:** LSTM for sequential patterns
- **Ensemble Stacking:** Meta-learning approaches
- **Explainable AI:** LIME/SHAP integration
- **Anomaly Detection:** Isolation Forest, Autoencoders
- **Web Dashboard:** Interactive fraud monitoring system

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Kaggle for providing the credit card fraud dataset
- Machine Learning Group of ULB for the original data collection
- Open source community for the excellent ML libraries

## 📞 Contact

- **Author:** Satyam Tiwari
- **GitHub:** [SatyamTiwari069](https://github.com/SatyamTiwari069)
- **Project Repository:** [Credit-Card-Fraud-Detection](https://github.com/SatyamTiwari069/Credit-Card-Fraud-Detection)

## 📚 References

1. [Credit Card Fraud Detection Dataset - Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
2. Pozzolo, A. D., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). Calibrating Probability with Undersampling for Unbalanced Classification.
3. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: synthetic minority oversampling technique.

---

⭐ **If you found this project helpful, please give it a star!** ⭐
