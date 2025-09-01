# Metaverse Transactions Anomaly Detection  

This project applies machine learning techniques to analyze a **metaverse transactions dataset** and classify anomalies into **Low Risk, Moderate Risk, and High Risk** categories. The workflow includes preprocessing, visualization, correlation analysis, feature scaling, and classification with multiple algorithms.  

---

## 🚀 Project Overview  

The main objectives of this project are:  

1. **Data Exploration & Preprocessing**  
   - Handling missing values  
   - Label encoding categorical features  
   - Feature scaling using MinMaxScaler  
   - Correlation analysis to reduce redundancy  

2. **Visualization**  
   - Pie chart & bar chart for class distribution  
   - Heatmap of feature correlations  
   - Confusion matrices for model evaluation  

3. **Classification Models**  
   - K-Nearest Neighbors (KNN)  
   - Decision Tree Classifier  
   - Naive Bayes (GaussianNB)  

4. **Evaluation**  
   - Accuracy scores  
   - Stratification comparison for Naive Bayes  
   - Confusion matrices  
   - Cross-validation scores  

---

## 📂 Dataset  

- The dataset is loaded from Google Drive:  
  ```
  /content/drive/MyDrive/metaverse_transactions_dataset_metaverse_transactions_dataset.csv
  ```
- Target variable: **`anomaly`**  
- Classes:  
  - `Low Risk`  
  - `Moderate Risk`  
  - `High Risk`  

---

## ⚙️ Installation & Requirements  

Clone this repository and install dependencies:  

```bash
git clone https://github.com/your-username/metaverse-anomaly-detection.git
cd metaverse-anomaly-detection
```

Install the required libraries:  

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 🧑‍💻 Usage  

Run the Jupyter Notebook or Python script in Google Colab:  

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

Make sure your dataset path is updated inside the script if you’re running locally.  

---

## 📊 Results  

- **Class Distribution**: Visualized via pie and bar charts.  
- **Feature Correlation**: Heatmap shows strong correlations, some features dropped.  
- **Classifier Accuracies**:  

| Classifier      | Accuracy (%) |
|-----------------|--------------|
| KNN             | ~XX% |
| Decision Tree   | ~XX% |
| Naive Bayes     | ~XX% |

- **Stratification Impact**: Stratified train-test split improves Naive Bayes performance.  
- **Confusion Matrices**: Generated for all classifiers to visualize misclassifications.  

---

## 📌 Key Learnings  

- Stratification in dataset splitting prevents biased results in imbalanced datasets.  
- Decision Trees and KNN handle categorical features differently than Naive Bayes.  
- Correlation analysis helps in reducing multicollinearity for better model performance.  

---

## 📸 Sample Visualizations  

### Class Distribution  
![Pie Chart of Anomalies](assets/class_distribution.png)  

### Confusion Matrix Example  
![Confusion Matrix](assets/confusion_matrix.png)  

---

## 🏆 Future Improvements  

- Implement ensemble methods (Random Forest, XGBoost).  
- Try deep learning models (LSTM/ANN) for anomaly detection.  
- Add ROC/AUC curves for multi-class evaluation.  
- Hyperparameter tuning for better accuracy.  

---

