import streamlit as st
import pickle
import json
import matplotlib.pyplot as plt
import pandas as pd

def plot_model_performance(results):
    """Plot accuracy, precision, recall, and F1 score as a bar chart"""
    # Transform the data to a DataFrame
    data = {
        'Model': [result['model'] for result in results],
        'Accuracy': [result['accuracy'] for result in results],
        'Precision': [result['precision'] for result in results],
        'Recall': [result['recall'] for result in results],
        'F1 Score': [result['f1_score'] for result in results]
    }

    df = pd.DataFrame(data).sort_values(by='Accuracy', ascending=False)

    # Plotting the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    df.set_index('Model').plot(kind='bar', ax=ax)
    plt.title('Model Performance Comparison', )
    plt.ylabel('Score')
    plt.xlabel('Metric')

    # Display the plot and data table in Streamlit
    st.pyplot(fig)

    st.write("### Model Performance Stats")
    st.dataframe(df)

    # Option to download the table as CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Full Analysis",
        csv,
        "model_performance_analysis.csv",
        "text/csv",
        key='download-csv'
    )
    
def load_model_and_results(model_path="model"):
    """Load the saved model, label encoder, and results"""
    with open(f"{model_path}/drug_side_effects_model.pkl", "rb") as f:
        model_dict = pickle.load(f)
        model = model_dict['model']
        label_encoder = model_dict['label_encoder']
     
    with open(f"{model_path}/all_model_results.json", "r") as f:
        results = json.load(f)
    
    return model, label_encoder, results

st.title("Model Performance Analysis")

model, _, results = load_model_and_results()
plot_model_performance(results)