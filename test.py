import pandas as pd
import numpy as np

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

# Load the chronic kidney disease dataset
file_path = 'chronic_kidney_disease.arff'
df = load_arff_to_dataframe(file_path)

if df is not None:
    print("Chronic Kidney Disease Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())
    print("\nDataset info:")
    print(df.info())
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    # Display basic statistics for numeric columns
    print("\nBasic statistics for numeric columns:")
    print(df.describe())
    
    # Display value counts for target variable
    print("\nTarget variable distribution:")
    print(df['class'].value_counts())
    
else:
    print("Failed to load the dataset.")