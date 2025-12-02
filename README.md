## K-Nearest Neighbors Classifier for Chronic Kidney Disease Prediction

CSCI 31022 - Machine Learning and Pattern Recognition  
Assignment 1: k-NN Classification

### Attributes:

1. **Age** (numerical) - age in years
2. **Blood Pressure** (numerical) - bp in mm/Hg
3. **Specific Gravity** (categorical) - sg (1.005,1.010,1.015,1.020,1.025)
4. **Albumin** (categorical) - al (0,1,2,3,4,5)
5. **Sugar** (categorical) - su (0,1,2,3,4,5)
6. **Red Blood Cells** (categorical) - rbc (normal,abnormal)
7. **Pus Cell** (categorical) - pc (normal,abnormal)
8. **Pus Cell Clumps** (categorical) - pcc (present,notpresent)
9. **Bacteria** (categorical) - ba (present,notpresent)
10. **Blood Glucose Random** (numerical) - bgr in mgs/dl
11. **Blood Urea** (numerical) - bu in mgs/dl
12. **Serum Creatinine** (numerical) - sc in mgs/dl
13. **Sodium** (numerical) - sod in mEq/L
14. **Potassium** (numerical) - pot in mEq/L
15. **Hemoglobin** (numerical) - hemo in gms
16. **Packed Cell Volume** (numerical) - pcv
17. **White Blood Cell Count** (numerical) - wbcc in cells/cumm
18. **Red Blood Cell Count** (numerical) - rbcc in millions/cmm
19. **Hypertension** (categorical) - htn (yes,no)
20. **Diabetes Mellitus** (categorical) - dm (yes,no)
21. **Coronary Artery Disease** (categorical) - cad (yes,no)
22. **Appetite** (categorical) - appet (good,poor)
23. **Pedal Edema** (categorical) - pe (yes,no)
24. **Anemia** (categorical) - ane (yes,no)
25. **Class** (target) - class (ckd,notckd)

- **Target Classes**:
  - `ckd`: Chronic Kidney Disease
  - `notckd`: No Chronic Kidney Disease



##  Machine Learning Pipeline

### 1. Data Loading and Exploration

- **File Format**: ARFF parser implementation
- **Data Analysis**: Shape, types, missing values, class distribution
- **Feature Identification**: Separate numerical and categorical features

### 2. Data Preprocessing

#### Categorical Features:

- **Missing Value Imputation**: Mode-based filling
- **Label Encoding**: Convert text to numerical values
- **Feature Count**: 13 categorical features processed

#### Numerical Features:

- **Missing Value Imputation**: Median-based strategy (robust to outliers)
- **Rationale**: Medical data often contains outliers and skewed distributions
- **Feature Count**: 11 numerical features processed

### 3. Data Cleaning

- **Class Balancing**: Remove classes with insufficient samples (<5 for cross-validation)
- **Final Dataset**: 399 samples with balanced class distribution
- **Quality Assurance**: Ensure all classes have adequate representation

### 4. Feature Scaling

- **Method**: StandardScaler (Z-score normalization)
- **Necessity**: KNN requires feature scaling for distance calculations
- **Formula**: `(feature - mean) / standard_deviation`
- **Result**: All 24 features scaled to mean=0, std=1

### 5. Data Splitting

- **Method**: Stratified train-test split
- **Ratio**: 80% training, 20% testing
- **Strategy**: Maintains class distribution in both sets
- **Random State**: 42 (for reproducibility)

### 6. Hyperparameter Tuning

- **Method**: 5-fold Cross-Validation
- **K Range**: 1 to 20 neighbors tested
- **Metric**: Accuracy score
- **Validation**: Robust performance estimation

### 7. Model Training and Evaluation

- **Algorithm**: K-Nearest Neighbors
- **Final Configuration**: Optimal K determined by cross-validation
- **Distance Metric**: Euclidean distance
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score

## 📈 Results and Performance

### Model Performance:

```
Final Model Results:
├── Optimal K Value: 1
├── Cross-Validation Accuracy: 98.75%
├── Test Accuracy: 97.50%
```

### Detailed Classification Report:

```
Final Model Performance (K=1):
Test Accuracy: 0.9750
Cross-Validation Accuracy: 0.9875

Classification Report:
              precision    recall  f1-score   support

           0       1.00      0.96      0.98        50
           2       0.94      1.00      0.97        30

    accuracy                           0.97        80
   macro avg       0.97      0.98      0.97        80
weighted avg       0.98      0.97      0.98        80


Confusion Matrix:
[[48  2]
 [ 0 30]]
```
