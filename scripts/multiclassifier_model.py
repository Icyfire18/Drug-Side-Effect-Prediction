import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import pickle
import json

# Step 1: Download and Load the SIDER Dataset
def load_data():
    side_effects_url = "http://sideeffects.embl.de/media/download/meddra_all_se.tsv.gz"
    drug_names_url = "http://sideeffects.embl.de/media/download/drug_names.tsv"

    side_effects_df = pd.read_csv(side_effects_url, sep='\t', compression='gzip',
                                  names=['drug_id', 'side_effect_id', 'meddra_type', 
                                         'frequency', 'placebo_frequency', 'side_effect_name'])
    drug_names_df = pd.read_csv(drug_names_url, sep='\t', names=['drug_id', 'drug_name'])
    return side_effects_df, drug_names_df

# Step 2: Preprocess the Data
def preprocess_data(side_effects_df, drug_names_df):
    data = side_effects_df.merge(drug_names_df, on='drug_id')
    data['presence'] = 1
    pivot_df = data.pivot_table(index='drug_name', columns='side_effect_name', 
                                values='presence', fill_value=0)
    pivot_df = pivot_df.loc[:, (pivot_df != 0).any(axis=0)]
    return pivot_df

# Step 3: Train a Multi-label Classification Model
def train_multiclass_model(pivot_df):
    # Prepare feature matrix X and labels y
    X = pivot_df.dropna(axis=1, how='all').values
    y = pivot_df.idxmax(axis=1).values  # Using the most frequent side effect as the label
    
    # First encode all labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Get the actual number of classes in the training data
    n_classes = len(np.unique(y_encoded))
    
    # Initialize and train XGBoost classifier with the correct number of classes
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=n_classes,
        eval_metric='mlogloss',
        use_label_encoder=False  # Add this parameter
    )
    
    # Convert labels to integer type
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    # Create DMatrix for XGBoost (optional but can help prevent issues)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Set up parameters
    params = {
        'objective': 'multi:softmax',
        'num_class': n_classes,
        'eval_metric': 'mlogloss',
        'max_depth': 6,
        'eta': 0.3,
        'min_child_weight': 1
    }
    
    # Train the model using the lower-level API
    num_rounds = 100
    model = xgb.train(params, dtrain, num_rounds)
    
    # Make predictions
    y_pred = model.predict(dtest)
    y_pred = y_pred.astype(int)
    
    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return model, y_test, y_pred, accuracy, report, label_encoder

def save_results(model, label_encoder, accuracy, report, save_path="model"):
    # Save model and label encoder
    model_dict = {
        'model': model,
        'label_encoder': label_encoder
    }
    
    with open(f"{save_path}/drug_side_effects_model.pkl", "wb") as f:
        pickle.dump(model_dict, f)
    
    # Save metrics and results
    results = {
        'accuracy': float(accuracy),
        'precision': float(report['weighted avg']['precision']),
        'recall': float(report['weighted avg']['recall']),
        'f1_score': float(report['weighted avg']['f1-score']),
        'support': int(report['weighted avg']['support']),
        'detailed_report': report
    }
    
    with open(f"{save_path}/model_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Model, label encoder, and results saved to {save_path}/")

def main():
    side_effects_df, drug_names_df = load_data()
    print("Data Loaded")
    
    print("Preprocessing data...")
    pivot_df = preprocess_data(side_effects_df, drug_names_df)

    print("Training the model...")
    model, y_test, y_pred, accuracy, report, label_encoder = train_multiclass_model(pivot_df)

    print("Saving Model")
    save_results(model, label_encoder, accuracy, report)
    
    # Decoding target labels to the original side effect names
    y_test_decoded = label_encoder.inverse_transform(y_test)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    
    print("\n--- Classification Results ---")
    print("\nOverall Performance Metrics:")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {report['weighted avg']['precision']:.3f}")
    print(f"Recall:    {report['weighted avg']['recall']:.3f}")
    print(f"F1-Score:  {report['weighted avg']['f1-score']:.3f}")
    print(f"Total samples (support): {int(report['weighted avg']['support'])}")

    if 1==2:
        print("\n--- Detailed Classification Results ---")
        print(f"Accuracy: {accuracy:.2f}")
        print("Classification Report:")
        print(classification_report(y_test_decoded, y_pred_decoded))
        
        print("\n--- Detailed Model Insights ---")
        print(f"Model Evaluation Metrics: {report}")

if __name__ == "__main__":
    main()

