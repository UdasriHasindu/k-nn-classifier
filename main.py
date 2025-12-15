# %%
import pandas as pd
import numpy as np

# %% [markdown]
# ### Data Loading and Exploration

# %%
def load_arff_to_dataframe(file_path):
    """
    Load ARFF file and convert it to a pandas DataFrame
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Find attribute definitions and data section
        attributes = []
        data_start_idx = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('@attribute'):
                # Extract attribute name (remove quotes if present)
                parts = line.split()
                attr_name = parts[1].replace("'", "").replace('"', '')
                attributes.append(attr_name)
            elif line.startswith('@data'):
                data_start_idx = i + 1
                break
        
        # Extract data rows
        data_rows = []
        for i in range(data_start_idx, len(lines)):
            line = lines[i].strip()
            if line and not line.startswith('%'):  # Skip empty lines and comments
                # Split by comma and clean each value
                row = [val.strip() if val.strip() != '?' else np.nan for val in line.split(',')]
                data_rows.append(row)
        
        # Ensure consistent row lengths - trim to match number of attributes
        for i, row in enumerate(data_rows):
            if len(row) > len(attributes):
                data_rows[i] = row[:len(attributes)]  # Trim extra columns
            elif len(row) < len(attributes):
                data_rows[i].extend([np.nan] * (len(attributes) - len(row)))  # Pad with NaN
        
        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=attributes)
        
        # Convert numeric columns to appropriate types
        numeric_cols = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except Exception as e:
        print(f"Error loading ARFF file: {e}")
        return None



# %%
# Load the chronic kidney disease dataset
file_path = 'chronic_kidney_disease.arff'
df = load_arff_to_dataframe(file_path)

df.head()

# %%
df.info()

# %%
print(f"\nMissing values:")
df.isnull().sum()

# %% [markdown]
# Separate numerical and categorical features

# %%
# check all datatypes of dataframe 
df.dtypes

# %%
numertic_cols = df.select_dtypes(include=['float64', 'int64']).columns
catergorical_cols = df.select_dtypes(include=['object']).columns

print(f"numertic_cols: {numertic_cols}")
print(f"catergorical_cols: {catergorical_cols}")

# %% [markdown]
# ### Preprocess
# 
# **Label encoding (categorical features)**
# 
# - which converts categorical text/string values into numerical values that machine learning algorithms can process.
# 

# %%
from sklearn.preprocessing import LabelEncoder

# %%
df.head()

# %%

# Encode categorical columns
df_processed = df.copy()
for col in catergorical_cols:
    
    if col in df_processed.columns:
        # filling missing values with mode
        col_mode = df_processed[col].mode()[0]
        df_processed[col] = df_processed[col].fillna(col_mode)

        # label encoding
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        

        
df_processed

# %% [markdown]
# **Note**
# 
# Using `median` is more accurate that the mean in this case
# 
# - Robust to outliers: Medical data often contains extreme values
# - Preserves distribution: Doesn't shift the data as much as mean
# - Appropriate for skewed data: Medical measurements are often not normally distributed

# %%
# fill missing numerical features
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')

numeric_features = df_processed[numertic_cols]
df_processed[numertic_cols] = imputer.fit_transform(numeric_features)

df_processed

# %%
df_processed.isnull().sum()

# %% [markdown]
# ### Data Cleaning
# 
# Remove classes with insufficient samples for cross-validation

# %%
# Check class distribution and clean data
print("Original class distribution:")
print(df_processed['class'].value_counts())

# Remove rare classes (less than 5 samples for 5-fold CV)
class_counts = df_processed['class'].value_counts()
min_samples = 5 

print(f"\nRemoving classes with fewer than {min_samples} samples...")
valid_classes = class_counts[class_counts >= min_samples].index

# Filter data
df_cleaned = df_processed[df_processed['class'].isin(valid_classes)].copy()

print(f"Classes removed: {set(class_counts.index) - set(valid_classes)}")
print(f"Final class distribution:")
print(df_cleaned['class'].value_counts())
print(f"Final dataset shape: {df_cleaned.shape}")

# %% [markdown]
# ### Split & Scale

# %%
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# %%
y = df_cleaned['class']
x = df_cleaned.iloc[:,:-1]

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.2, 
    stratify=y,  
    random_state=42
)

print(f"Training set shape: {x_train.shape}")
print(f"Test set shape: {x_test.shape}")
print(f"Training class distribution:")
print(pd.Series(y_train).value_counts())

# %% [markdown]
# ### Tuning & Training

# %%
# Hyperparameter Tuning (Fixed)
from sklearn.neighbors import KNeighborsClassifier

# Check class distribution first
print("Class distribution in training set:")
print(pd.Series(y_train).value_counts())

# Now we can safely use 5-fold CV since all classes have ≥5 samples
cv_folds = 5
print(f"Using {cv_folds}-fold cross-validation")

cv_score = []

for k in range(1, 21):  # Test K from 1 to 20
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_train, y_train, cv=cv_folds, scoring='accuracy')
    cv_score.append(scores.mean())
    
    # Print progress for first 10 and every 5th
    if k <= 10 or k % 5 == 0:
        print(f"K={k}: CV Accuracy = {scores.mean():.4f} (±{scores.std():.4f})")

print(f"\nCV Scores: {cv_score}")
best_score = max(cv_score)
best_k = cv_score.index(best_score) + 1

print(f"\nBest K: {best_k}")
print(f"Best CV Accuracy: {best_score:.4f}")

# %% [markdown]
# Visualize K vs Accuracy

# %%

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), cv_score, 'bo-', markersize=6)
plt.axvline(x=best_k, color='r', linestyle='--', alpha=0.7, label=f'Best K = {best_k}')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('KNN Hyperparameter Tuning: K vs Accuracy')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# Evaluate model

# %%
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Train the final model with best K
final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(x_train, y_train)



# %%
# Make predictions
y_pred = final_knn.predict(x_test)

# Calculate accuracy
test_accuracy = accuracy_score(y_test, y_pred)

print(f"Final Model Performance (K={best_k}):")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Cross-Validation Accuracy: {best_score:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# %%



