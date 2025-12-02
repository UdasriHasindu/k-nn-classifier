"""
KNN Classifier for Chronic Kidney Disease Prediction
==================================================

This implementation demonstrates a complete KNN classifier pipeline with the following steps:
1. Data Loading and Exploration
2. Data Preprocessing and Cleaning
3. Feature Engineering and Encoding
4. Data Splitting
5. Feature Scaling
6. Model Training and Hyperparameter Tuning
7. Model Evaluation
8. Visualization of Results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class KNNChronicKidneyClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = None
        
    def load_arff_to_dataframe(self, file_path):
        """
        STEP 1: DATA LOADING
        Load ARFF file and convert it to a pandas DataFrame
        """
        print("=" * 60)
        print("STEP 1: LOADING DATA")
        print("=" * 60)
        
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
            
            print(f"✓ Dataset loaded successfully!")
            print(f"✓ Shape: {df.shape}")
            print(f"✓ Features: {len(df.columns)-1}, Target: 1 (class)")
            
            return df
            
        except Exception as e:
            print(f"✗ Error loading ARFF file: {e}")
            return None
    
    def explore_data(self, df):
        """
        STEP 2: DATA EXPLORATION
        Analyze the dataset structure and patterns
        """
        print("\n" + "=" * 60)
        print("STEP 2: DATA EXPLORATION")
        print("=" * 60)
        
        print(f"\nDataset Overview:")
        print(f"- Total samples: {len(df)}")
        print(f"- Total features: {len(df.columns)-1}")
        print(f"- Target variable: {df.columns[-1]}")
        
        print(f"\nFeature types:")
        print(df.dtypes)
        
        print(f"\nMissing values:")
        missing_data = df.isnull().sum()
        print(missing_data[missing_data > 0])
        
        print(f"\nTarget distribution:")
        target_dist = df['class'].value_counts()
        print(target_dist)
        print(f"Class balance: {target_dist.min()/target_dist.max():.2f}")
        
        # Identify categorical and numerical columns
        categorical_cols = []
        numerical_cols = []
        
        for col in df.columns[:-1]:  # Exclude target
            if df[col].dtype == 'object':
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        
        print(f"\nCategorical features ({len(categorical_cols)}): {categorical_cols}")
        print(f"Numerical features ({len(numerical_cols)}): {numerical_cols}")
        
        return categorical_cols, numerical_cols
    
    def preprocess_data(self, df, categorical_cols, numerical_cols):
        """
        STEP 3: DATA PREPROCESSING
        Handle missing values and encode categorical variables
        """
        print("\n" + "=" * 60)
        print("STEP 3: DATA PREPROCESSING")
        print("=" * 60)
        
        df_processed = df.copy()
        
        # Handle categorical variables
        print(f"\nEncoding categorical variables...")
        for col in categorical_cols:
            if col in df_processed.columns:
                # Fill missing values with mode
                mode_value = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'unknown'
                df_processed[col].fillna(mode_value, inplace=True)
                
                # Label encode
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                self.label_encoders[col] = le
                print(f"✓ {col}: {len(le.classes_)} unique values")
        
        # Handle target variable - fix class imbalance issue
        print(f"\nHandling target variable...")
        print(f"Original class distribution:")
        print(df_processed['class'].value_counts())
        
        # Remove classes with very few samples (less than 2)
        class_counts = df_processed['class'].value_counts()
        classes_to_keep = class_counts[class_counts >= 2].index
        df_processed = df_processed[df_processed['class'].isin(classes_to_keep)]
        
        print(f"After removing rare classes:")
        print(df_processed['class'].value_counts())
        
        # Encode target variable
        le_target = LabelEncoder()
        df_processed['class'] = le_target.fit_transform(df_processed['class'])
        self.label_encoders['class'] = le_target
        print(f"✓ class encoded: {le_target.classes_}")
        
        # Handle numerical variables (imputation will be done later)
        print(f"\nNumerical variables will be imputed during preprocessing...")
        
        print(f"\n✓ Preprocessing completed!")
        print(f"✓ Final shape: {df_processed.shape}")
        
        return df_processed
    
    def split_and_scale_data(self, df_processed):
        """
        STEP 4: DATA SPLITTING AND FEATURE SCALING
        Split data into train/test and scale features
        """
        print("\n" + "=" * 60)
        print("STEP 4: DATA SPLITTING AND FEATURE SCALING")
        print("=" * 60)
        
        # Separate features and target
        X = df_processed.drop('class', axis=1)
        y = df_processed['class']
        
        self.feature_names = X.columns.tolist()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Handle missing values with median imputation
        print(f"\nHandling missing values...")
        X_train_imputed = self.imputer.fit_transform(X_train)
        X_test_imputed = self.imputer.transform(X_test)
        
        # Scale the features
        print(f"Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        X_test_scaled = self.scaler.transform(X_test_imputed)
        
        print(f"✓ Data preprocessing completed!")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def find_best_k(self, X_train, y_train):
        """
        STEP 5: HYPERPARAMETER TUNING
        Find the optimal value of K using cross-validation
        """
        print("\n" + "=" * 60)
        print("STEP 5: HYPERPARAMETER TUNING")
        print("=" * 60)
        
        # Test different values of K
        k_range = range(1, min(31, len(X_train)//5))  # Avoid K too large
        cv_scores = []
        
        print(f"Testing K values from 1 to {max(k_range)}...")
        
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
            cv_scores.append(scores.mean())
            if k <= 10 or k % 5 == 0:  # Print progress for first 10 and every 5th
                print(f"K={k}: CV Accuracy = {scores.mean():.4f} (±{scores.std():.4f})")
        
        # Find best K
        best_k = k_range[np.argmax(cv_scores)]
        best_score = max(cv_scores)
        
        print(f"\n✓ Best K: {best_k}")
        print(f"✓ Best CV Accuracy: {best_score:.4f}")
        
        # Plot K vs Accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, cv_scores, 'bo-', markersize=6)
        plt.axvline(x=best_k, color='r', linestyle='--', alpha=0.7, label=f'Best K = {best_k}')
        plt.xlabel('Number of Neighbors (K)')
        plt.ylabel('Cross-Validation Accuracy')
        plt.title('KNN Hyperparameter Tuning: K vs Accuracy')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        return best_k
    
    def train_model(self, X_train, y_train, best_k):
        """
        STEP 6: MODEL TRAINING
        Train the final KNN model with the best K
        """
        print("\n" + "=" * 60)
        print("STEP 6: MODEL TRAINING")
        print("=" * 60)
        
        # Train the model with best K
        self.model = KNeighborsClassifier(n_neighbors=best_k)
        self.model.fit(X_train, y_train)
        
        print(f"✓ KNN model trained with K={best_k}")
        print(f"✓ Training samples: {X_train.shape[0]}")
        print(f"✓ Features: {X_train.shape[1]}")
    
    def evaluate_model(self, X_train, X_test, y_train, y_test):
        """
        STEP 7: MODEL EVALUATION
        Evaluate the model performance on train and test sets
        """
        print("\n" + "=" * 60)
        print("STEP 7: MODEL EVALUATION")
        print("=" * 60)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Calculate accuracies
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Testing Accuracy: {test_accuracy:.4f}")
        print(f"Generalization Gap: {train_accuracy - test_accuracy:.4f}")
        
        # Detailed classification report
        print(f"\nDetailed Classification Report (Test Set):")
        target_names = self.label_encoders['class'].classes_
        print(classification_report(y_test, y_test_pred, target_names=target_names))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred)
        
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Confusion Matrix
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Plot 2: ROC Curve (for binary classification)
        plt.subplot(1, 2, 2)
        if len(np.unique(y_test)) == 2:
            # Get prediction probabilities
            y_test_proba = self.model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_test_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'ROC Curve\n(Multi-class)', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
        return test_accuracy
    
    def predict_new_sample(self, sample_data):
        """
        Predict class for new sample(s)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet!")
        
        # Preprocess the sample
        sample_processed = self.scaler.transform(sample_data.reshape(1, -1))
        
        # Make prediction
        prediction = self.model.predict(sample_processed)[0]
        probability = self.model.predict_proba(sample_processed)[0]
        
        # Decode the prediction
        predicted_class = self.label_encoders['class'].inverse_transform([prediction])[0]
        
        return predicted_class, probability
    
    def run_complete_pipeline(self, file_path):
        """
        Run the complete KNN classification pipeline
        """
        print("🩺 CHRONIC KIDNEY DISEASE PREDICTION WITH KNN CLASSIFIER")
        print("=" * 60)
        
        # Step 1: Load data
        df = self.load_arff_to_dataframe(file_path)
        if df is None:
            return None
        
        # Step 2: Explore data
        categorical_cols, numerical_cols = self.explore_data(df)
        
        # Step 3: Preprocess data
        df_processed = self.preprocess_data(df, categorical_cols, numerical_cols)
        
        # Step 4: Split and scale data
        X_train, X_test, y_train, y_test = self.split_and_scale_data(df_processed)
        
        # Step 5: Find best K
        best_k = self.find_best_k(X_train, y_train)
        
        # Step 6: Train model
        self.train_model(X_train, y_train, best_k)
        
        # Step 7: Evaluate model
        test_accuracy = self.evaluate_model(X_train, X_test, y_train, y_test)
        
        print(f"\n" + "=" * 60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"✓ Final Model: KNN with K={best_k}")
        print(f"✓ Test Accuracy: {test_accuracy:.4f}")
        print(f"✓ Model ready for predictions!")
        
        return self.model, test_accuracy

# Run the complete pipeline
if __name__ == "__main__":
    # Initialize the classifier
    knn_classifier = KNNChronicKidneyClassifier()
    
    # Run the complete pipeline
    file_path = 'chronic_kidney_disease.arff'
    model, accuracy = knn_classifier.run_complete_pipeline(file_path)