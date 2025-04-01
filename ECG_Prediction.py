import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
import nbimporter
from Ecg import predict_ecg_file



def getData(name):
    # paths to the csv and anotation files
    
    path_csv = r'mitbih_database\{}.csv'.format(name)
    path_anotation = r'mitbih_database\{}annotations.txt'.format(name)
    # colums of the csv and anotation files
    colmns = ['Time', 'Sample', 'Type', 'Sub', 'Chan', 'Num']
    csv_columns = ['Sample', "MILI", 'V5']
    column_types = {
    'Sample': float,
    'MLII': float,
    'V5': int
}
    # anomleis and their numaric value
    # anomlie = {'N': 0, 'L': -0.5, 'R': 0.5, 'A': -1, 'V':-1}
    # decarator function to run the functions that reads the data and louds it one epoch at a time 
    csv_data = pd.read_csv(path_csv, sep=r'\s*,\s*', names=csv_columns, header=0, encoding='ascii', engine='python')
    # csv_data.drop(0, inplace=True)
    ann_data = pd.read_csv(path_anotation, delimiter='\s+')
    # ann_data = ann_data.reindex(cols)
    return ann_data, csv_data



def create_image_ecg(data, data_a, name):
    # shows a section of the ecg with the anomalies on the matplotlib plot
    fs = 360
    Ts = 1/fs

    labels = data_a['Type']
    
    data['Time'] = data['Sample']*Ts
    
    plt.plot(data['time'], data['V5'])

    for i, label in enumerate(labels):
        plt.annotate(label, (data_a['Sample'][i]*Ts, data['V5'][data_a['Sample'][i]]))
                 
    plt.plot(data_a['Sample']*Ts, data['V5'].iloc[data_a['Sample']], label=data_a['Type'], marker='*', linestyle='None')
    plt.xlim((0, 20))
    plt.grid()
    plt.xlabel('time (ms)')
    plt.ylabel('ecg')
    plt.title(f'ECG_{name}')
    plt.show()
     
def image_plotly(data, data_a, name):
    # shows the ecg with the anomalies on the plot
    fs = 360
    Ts = 1/fs
    data['Sample'] = pd.to_numeric(data['Sample'], errors='coerce')
    data['time'] = data['Sample']*Ts
    data['V5'] = data['V5'] - 980
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['time'], y=data['V5'], mode='lines', name='V5', line=dict(color='blue')))
    fig.update_layout(xaxis_title='time (ms)', yaxis_title='ecg')
    fig.update_xaxes(rangeslider_visible=True)
    fig.show()

def diagnose_by_symptoms(symptoms, ecg_findings=None):
    """
    Diagnose potential cardiac conditions based on symptom inputs and optional ECG findings.
    
    Parameters:
    - symptoms (dict): Dictionary of symptoms with boolean or severity values.
      Expected keys: 'palpitations', 'chest_pain', 'dyspnea', 'syncope',
                     'fatigue', 'dizziness', 'edema'
    - ecg_findings (dict): Optional dictionary containing ECG findings (e.g., 'LBBB', 'RBBB', 'PACs', 'PVCs').
    
    Returns:
    - diagnosis (dict): Dictionary with probable diagnoses and tiered probabilities.
    """
    diagnosis = {
        "Primary": [],
        "Secondary": [],
        "Tertiary": []
    }
    
    # Example logic for each symptom:
    # 1. Palpitations
    if symptoms.get('palpitations', False):
        # Primary: Benign ectopic beats (PACs/PVCs)
        diagnosis["Primary"].append("Premature Atrial/Ventricular Beats")
        # Secondary: Paroxysmal Atrial Fibrillation
        diagnosis["Secondary"].append("Paroxysmal Atrial Fibrillation (AF)")
        # Tertiary: Supraventricular Tachycardia (SVT)
        diagnosis["Tertiary"].append("Supraventricular Tachycardia (SVT)")
    
    # 2. Chest Pain
    if symptoms.get('chest_pain', False):
        # Primary: Ischemia due to Coronary Artery Disease (CAD)
        diagnosis["Primary"].append("Coronary Artery Disease (Ischemia)")
        # Secondary: Acute Myocardial Infarction (especially if new LBBB on ECG)
        if ecg_findings and ecg_findings.get('LBBB', False):
            diagnosis["Secondary"].append("Acute Myocardial Infarction (AMI)")
        else:
            diagnosis["Secondary"].append("Acute Myocardial Infarction (if other risk factors present)")
        # Tertiary: Pericarditis or nonischemic causes
        diagnosis["Tertiary"].append("Pericarditis or Other Nonischemic Causes")
    
    # 3. Dyspnea (Shortness of Breath)
    if symptoms.get('dyspnea', False):
        # Primary: Heart Failure/Cardiomyopathy
        diagnosis["Primary"].append("Heart Failure / Cardiomyopathy")
        # Secondary: Ischemic Heart Disease
        diagnosis["Secondary"].append("Ischemic Heart Disease")
        # Tertiary: Pulmonary causes (e.g., COPD, pulmonary embolism)
        diagnosis["Tertiary"].append("Pulmonary Causes (e.g., COPD, Pulmonary Embolism)")
    
    # 4. Syncope or Presyncope
    if symptoms.get('syncope', False):
        # Primary: Arrhythmia-related syncope (e.g., high-grade AV block, VT)
        diagnosis["Primary"].append("Arrhythmia-related Syncope (e.g., Ventricular Tachycardia, High-Grade AV Block)")
        # Secondary: Bradyarrhythmia
        diagnosis["Secondary"].append("Bradyarrhythmia")
        # Tertiary: Orthostatic Hypotension / Vasovagal episodes
        diagnosis["Tertiary"].append("Orthostatic Hypotension / Vasovagal Episodes")
    
    # 5. Fatigue / Exercise Intolerance
    if symptoms.get('fatigue', False):
        # Primary: Heart Failure
        diagnosis["Primary"].append("Heart Failure")
        # Secondary: Ischemic Heart Disease
        diagnosis["Secondary"].append("Ischemic Heart Disease")
        # Tertiary: Non-cardiac causes (e.g., Anemia, Thyroid Dysfunction)
        diagnosis["Tertiary"].append("Non-Cardiac Causes (e.g., Anemia, Thyroid Dysfunction)")
    
    # 6. Dizziness or Lightheadedness
    if symptoms.get('dizziness', False):
        # Primary: Arrhythmias (including tachy- or bradyarrhythmias)
        diagnosis["Primary"].append("Arrhythmias (e.g., Ventricular Tachycardia, Severe Bradycardia)")
        # Secondary: Conduction abnormalities (e.g., intermittent LBBB/RBBB)
        diagnosis["Secondary"].append("Conduction Abnormalities (e.g., LBBB, RBBB)")
        # Tertiary: Medication effects or dehydration
        diagnosis["Tertiary"].append("Medication Effects / Dehydration")
    
    # 7. Peripheral Edema
    if symptoms.get('edema', False):
        # Primary: Heart Failure (especially right-sided)
        diagnosis["Primary"].append("Right-Sided Heart Failure")
        # Secondary: Renal Dysfunction leading to fluid retention
        diagnosis["Secondary"].append("Renal Dysfunction")
        # Tertiary: Venous Insufficiency
        diagnosis["Tertiary"].append("Venous Insufficiency")
    
    # Optionally, incorporate ECG findings into the logic:
    if ecg_findings:
        if ecg_findings.get('LBBB', False):
            diagnosis["Primary"].append("Underlying Structural Heart Disease (LBBB)")
        if ecg_findings.get('RBBB', False):
            # RBBB in isolation may be benign, but consider context:
            diagnosis["Secondary"].append("Possible Conduction Delay (RBBB)")
        if ecg_findings.get('PACs', 0) > 30:
            diagnosis["Primary"].append("High Atrial Ectopy - Increased AF Risk")
        if ecg_findings.get('PVCs', 0) > 10:  # assuming percentage or threshold value
            diagnosis["Primary"].append("High Ventricular Ectopy - Risk of PVC-induced Cardiomyopathy")
    
    return diagnosis

# Example usage:
if __name__ == "__main__":
    # Sample symptom input (adjust booleans or values as needed)
    symptoms = {
        "palpitations": True,
        "chest_pain": True,
        "dyspnea": True,
        "syncope": False,
        "fatigue": True,
        "dizziness": True,
        "edema": False
    }
    
    # Sample ECG findings: For instance, LBBB present, 35 PACs per hour, and 12% PVCs.
    ecg_findings = {
        "LBBB": True,
        "RBBB": False,
        "PACs": 35,
        "PVCs": 12
    }
    
    results = diagnose_by_symptoms(symptoms, ecg_findings)
    print("Diagnostic Recommendations:")
    for tier, diagnoses in results.items():
        print(f"\n{tier} Diagnoses:")
        for diag in diagnoses:
            print(f" - {diag}")


def main():
    ann, csv = getData("100")
    
    image_plotly(csv, ann, "100")
    
    print(predict_ecg_file("100"))


if __name__ == '__main__':
    main()