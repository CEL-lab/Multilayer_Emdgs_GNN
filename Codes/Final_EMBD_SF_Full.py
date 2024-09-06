import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed
import joblib

# File paths
raw_data_file_path = 'Incidents.xlsx'
embedding_file_path = '/mmfs1/home/muhammad.kazim/Embeddings_EMGNN_Paper_Codes/Multiplex_NW_Emd/augmented_data_with_embeddings_c.csv'
embedding_with_features_file_path = '/mmfs1/home/muhammad.kazim/Embeddings_EMGNN_Paper_Codes/Multiplex_NW_Emd/augmented_data_with_embeddings_and_features.csv'
output_file_path = '/mmfs1/home/muhammad.kazim/Embeddings_EMGNN_Paper_Codes/Multiplex_NW_Emd/model_comparison_OGE_Causes_corrected.txt'

# Load datasets once
raw_data = pd.read_excel(raw_data_file_path, engine='openpyxl')
embedding_data = pd.read_csv(embedding_file_path)
embedding_with_features_data = pd.read_csv(embedding_with_features_file_path)

# Ensure 'Job Substation' in raw_data matches the embedding keys
raw_data['Job Substation'] = raw_data['Job Substation'].str.replace(' ', '_')

# Full features from raw data
full_features = raw_data.columns.tolist()

# Selected 7 features
selected_features = ['Job Region', 'Month/Day/Year', 'Custs Affected', 'OGE Causes', 'Major Storm Event  Y (Yes) or N (No)', 'Distribution, Substation, Transmission']

# Function to enforce consistent data types
def enforce_consistent_data_types(df):
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
    for col in df.select_dtypes(include=['datetime']).columns:
        df[col] = df[col].astype('datetime64[ns]')
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Apply consistent data type enforcement
raw_data = enforce_consistent_data_types(raw_data)
embedding_with_features_data = enforce_consistent_data_types(embedding_with_features_data)

# Prepare data for the embedding dataset
def prepare_embedding_data(data, target_column):
    X = data.iloc[:, 1:].values  # All columns except the first one (target column)
    y = data[target_column].values
    return X, y

# Prepare data for the raw dataset
def prepare_raw_data(data, features, target_column):
    X = data[features].drop(columns=target_column, errors='ignore')
    y = data[target_column]
    return X, y

# Encode labels
def encode_labels(y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le

# Perform 5-fold cross-validation with multiple models
def perform_cv(X, y, models, cv=5, use_gpu=False):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    results = {}

    def process_model(name, model):
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=[np.number, 'datetime64']).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]), numeric_features),
                ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
            ]
        )

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        scores = cross_val_score(pipeline, X, y, cv=kf, scoring='accuracy', n_jobs=1)  # Limit parallelism
        return name, (scores.mean(), scores.std())

    results = Parallel(n_jobs=2)(delayed(process_model)(name, model) for name, model in models.items())
    return dict(results)

# Predict using embeddings and raw data
def predict_from_data(data, target_column, models):
    # Prepare and encode data
    X, y = prepare_embedding_data(data, target_column)
    y_encoded, le = encode_labels(y)

    # Perform cross-validation
    results = perform_cv(pd.DataFrame(X), y_encoded, models, cv=5)

    return results

def predict_from_raw_data(data, features, target_column, models):
    # Prepare and encode data
    X, y = prepare_raw_data(data, features, target_column)
    y_encoded, le = encode_labels(y)

    # Perform cross-validation
    results = perform_cv(X, y_encoded, models, cv=5)

    return results

# New function to handle embeddings and additional features
def predict_from_embedding_and_features(data, target_column, models):
    # Prepare and encode data (all features including embeddings)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    y_encoded, le = encode_labels(y)

    # Perform cross-validation
    results = perform_cv(X, y_encoded, models, cv=5)

    return results

# Models to compare
models = {
    'RandomForest': RandomForestClassifier(n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(),
    'KNeighbors': KNeighborsClassifier(n_jobs=-1)
}

# Perform prediction and comparison

results = []
for target in ['Job Area (DISTRICT)']:
    results.append(f"Target: {target}\n")
    
    # Prediction using full raw data
    full_data_results = predict_from_raw_data(raw_data, full_features, target, models)
    results.append("Full Raw Data Results:\n")
    for model, (mean_acc, std_acc) in full_data_results.items():
        results.append(f"{model}: Mean Accuracy = {mean_acc:.4f}, Standard Deviation = {std_acc:.4f}\n")
    
    # Prediction using raw data with selected features
    selected_results = predict_from_raw_data(raw_data, selected_features, target, models)
    results.append("Raw Data with Selected Features Results:\n")
    for model, (mean_acc, std_acc) in selected_results.items():
        results.append(f"{model}: Mean Accuracy = {mean_acc:.4f}, Standard Deviation = {std_acc:.4f}\n")
    
    # Prediction using embedding data
    embedding_results = predict_from_data(embedding_data, target, models)
    results.append("Embedding Data Results:\n")
    for model, (mean_acc, std_acc) in embedding_results.items():
        results.append(f"{model}: Mean Accuracy = {mean_acc:.4f}, Standard Deviation = {std_acc:.4f}\n")
    
    # New case: Prediction using embeddings and all other features
    embedding_with_features_results = predict_from_embedding_and_features(embedding_with_features_data, target, models)
    results.append("Embedding and Full Features Results:\n")
    for model, (mean_acc, std_acc) in embedding_with_features_results.items():
        results.append(f"{model}: Mean Accuracy = {mean_acc:.4f}, Standard Deviation = {std_acc:.4f}\n")
    
    results.append("\n")

# Write results to output file
with open(output_file_path, 'w') as f:
    f.writelines(results)

print("Prediction results have been written to:", output_file_path)
