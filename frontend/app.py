import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json


def load_model_and_results(model_path="model"):
    """Load the saved model, label encoder, and results"""
    with open(f"{model_path}/drug_side_effects_lightGBM_model.pkl", "rb") as f:
        model_dict = pickle.load(f)
        model = model_dict['model']
        label_encoder = model_dict['label_encoder']
     
    with open(f"{model_path}/all_model_results.json", "r") as f:
        results = json.load(f)
    
    return model, label_encoder, results

@st.cache_data
def load_data():
    """Load the drug names, side effects, and disease data"""
    side_effects_url = "http://sideeffects.embl.de/media/download/meddra_all_se.tsv.gz"
    drug_names_url = "http://sideeffects.embl.de/media/download/drug_names.tsv"
    disease_data_url = "https://www.kaggle.com/api/v1/datasets/download/manncodes/drug-prescription-to-disease-dataset?dataset_version_number=1"

    side_effects_df = pd.read_csv(
        side_effects_url,
        sep='\t',
        compression='gzip',
        names=[
            'compound_id_flat', 
            'compound_id_stereo', 
            'label_umls_id', 
            'meddra_type', 
            'meddra_umls_id', 
            'side_effect_name'
        ]
    )
    drug_names_df = pd.read_csv(drug_names_url, sep='\t', names=['compound_id_flat', 'drug_name'])
    disease_data_df = pd.read_csv(disease_data_url, sep=',', compression='zip')[['disease', 'drug']]

    return side_effects_df, drug_names_df, disease_data_df

def get_drug_side_effects(drug_name, side_effects_df, drug_names_df):
    """Get side effects for a specific drug"""
    compound_id_flat = drug_names_df[drug_names_df['drug_name'] == drug_name]['compound_id_flat'].values[0]
    drug_ses = side_effects_df[side_effects_df['compound_id_flat'] == compound_id_flat]['side_effect_name'].tolist()
    return drug_ses

def create_feature_vector(drug1_ses, drug2_ses, all_side_effects):
    """Create a feature vector for the drug combination"""
    feature_vector = np.zeros(len(all_side_effects))
    for se in drug1_ses.union(drug2_ses):
        if se in all_side_effects:
            idx = all_side_effects.index(se)
            feature_vector[idx] = 1
    return feature_vector

def predict_drug_combinations(drug1, model, label_encoder, side_effects_df, drug_names_df, drugs_to_be_selected_from):
    """Predict safety scores for combining drug1 with all other drugs"""
    # Get all unique side effects for feature vector creation
    all_side_effects = side_effects_df['side_effect_name'].unique().tolist()
    
    # Get side effects for the first drug
    drug1_ses = set(get_drug_side_effects(drug1, side_effects_df, drug_names_df))
    
    # Get all available drugs except drug1
    available_drugs = drug_names_df['drug_name'].unique().tolist()
    other_drugs = [drug for drug in available_drugs if drug != drug1 if drug in drugs_to_be_selected_from]
    
    # Calculate combinations and predictions
    combinations = []
    for drug2 in other_drugs:
        drug2_ses = set(get_drug_side_effects(drug2, side_effects_df, drug_names_df))
        
        # Calculate common side effects and risk score
        common_ses = drug1_ses.intersection(drug2_ses)
        risk_score = len(common_ses) / (len(drug1_ses) + len(drug2_ses)) * 100
        
        # Create feature vector for model prediction
        feature_vector = create_feature_vector(drug1_ses, drug2_ses, all_side_effects)
        
        # Get model prediction
        dmatrix  = feature_vector.reshape(1, -1)
        prediction = model.predict(dmatrix)[0]
        severity_score = float(prediction) / len(label_encoder.classes_) * 100
        
        # Combined risk score (weighted average of overlap and model prediction)
        combined_score = 0.6 * risk_score + 0.4 * severity_score
        
        combinations.append({
            'drug': drug2,
            'risk_score': risk_score,
            'severity_score': severity_score,
            'combined_score': combined_score,
            'common_side_effects': common_ses
        })
    
    # Sort combinations by combined score
    return sorted(combinations, key=lambda x: x['combined_score'])



def drug_recommendation_app():
    
    # Load model, data, and results
    model, label_encoder, _ = load_model_and_results()
    side_effects_df, drug_names_df, disease_data_df = load_data()
    disease_data_df = disease_data_df.drop_duplicates() # Remove duplicates
    merged_df = side_effects_df.merge(drug_names_df, on='compound_id_flat').merge(disease_data_df, left_on='drug_name', right_on='drug')
    
    # Main content
    st.write("Select diseases and drugs to find the safest combinations")
    available_diseases = disease_data_df['disease'].unique().tolist()
    
    # Disease and drug selection
    disease1 = st.selectbox("Select First Disease", available_diseases)

    if disease1:
        filtered_drugs = merged_df[merged_df['disease'] == disease1]['drug_name'].unique().tolist()
        drug1 = st.selectbox("Select Drug for First Disease", filtered_drugs)
    
        disease2 = st.selectbox("Select Second Disease", available_diseases)
    
        if drug1 and disease2:
            st.write("### Selected Drug Side Effects")
            drug1_ses = get_drug_side_effects(drug1, side_effects_df, drug_names_df)
            with st.expander("View Side Effects"):
                st.write(", ".join(drug1_ses))
            
            # Button for prediction
            if st.button("Predict Drug"):

                drugs_to_be_selected_from = merged_df[merged_df['disease'] == disease2]['drug_name'].unique().tolist()
                # Analyze combinations
                combinations = predict_drug_combinations(
                    drug1, model, label_encoder, side_effects_df, drug_names_df, drugs_to_be_selected_from
                )
                
                st.write("### Recommended Drug Combinations for Second Disease")
                st.write("Sorted by predicted safety (lowest risk first)")
                
                # Display top 2 safest combinations
                for i, combo in enumerate(combinations[:2]):
                    with st.container():
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.subheader(f"{i+1}. {combo['drug']}")
                        
                        with col2:
                            risk_color = 'green' if combo['combined_score'] < 30 else 'orange' if combo['combined_score'] < 60 else 'red'
                            st.markdown(f"Risk Score: <span style='color:{risk_color}'>{combo['combined_score']:.1f}%</span>", unsafe_allow_html=True)
                        
                        with col3:
                            st.write(f"Common Effects: {len(combo['common_side_effects'])}")
                        
                        # Show detailed information in an expander
                        with st.expander("View Details"):
                            st.write("Risk Breakdown:")
                            st.write(f"- Overlap Risk: {combo['risk_score']:.1f}%")
                            st.write(f"- Severity Risk: {combo['severity_score']:.1f}%")
                            st.write("\nCommon Side Effects:")
                            if combo['common_side_effects']:
                                st.write(", ".join(combo['common_side_effects']))
                            else:
                                st.write("No common side effects found")
                        
                        st.divider()
                
                # Add download button for full results
                df_combinations = pd.DataFrame(combinations)
                csv = df_combinations.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Full Analysis",
                    csv,
                    "drug_combinations.csv",
                    "text/csv",
                    key='download-csv'
                )

def main():
    st.set_page_config(layout="wide")
    st.title("Drug Side Effects Analysis")

    try:
        drug_recommendation_app()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please make sure the model and data files are properly loaded.")


if __name__ == "__main__":
    main()
