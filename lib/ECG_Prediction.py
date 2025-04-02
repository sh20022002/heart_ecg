import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
from Ecg import predict_ecg_file




import pandas as pd

def getData(file):
    """
    Reads the ECG CSV file and returns a DataFrame.
    This function cleans the column names by removing extra quotes and whitespace,
    then renames the columns to a fixed set:
      - First column is renamed to "Sample"
      - Second column is renamed to "MLII"
      - Third column is renamed to "V5"
      
    Parameters:
    - file: A file path or file-like object containing the ECG CSV data.
    
    Returns:
    - df: A DataFrame with standardized column names.
    """
    file.seek(0)  # Reset file pointer to the beginning
    try:
        df = pd.read_csv(file, engine='python')
    except Exception as e:
        print("Error reading file:", e)
        raise

    # Clean column names: remove extra quotes and whitespace.
    df.columns = df.columns.str.replace("'", "", regex=False).str.strip()
    print("Cleaned DataFrame columns:", df.columns)

    # Rename columns according to a fixed mapping if there are exactly three columns.
    if len(df.columns) == 3:
        df.columns = ['Sample', 'MLII', 'V5']
    else:
        # Alternatively, rename only the first and last columns if your file structure varies.
        new_names = list(df.columns)
        if len(new_names) >= 1:
            new_names[0] = "Sample"
        if len(new_names) >= 3:
            new_names[2] = "V5"
        # You can set new_names for other columns as needed, e.g. second column to "MLII"
        if len(new_names) >= 2:
            new_names[1] = "MLII"
        df.columns = new_names

    print("Final DataFrame columns:", df.columns)
    return df


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
     


def image_plotly(file):
    """
    Create an interactive Plotly plot for ECG data with two leads: V5 and MLII.

    Parameters:
    - file: File path or file-like object containing the ECG CSV data.

    Returns:
    - fig: A Plotly figure object.
    """
    data = getData(file)
    fs = 360
    Ts = 1 / fs

    # Ensure 'Sample' is numeric
    data['Sample'] = pd.to_numeric(data['Sample'], errors='coerce')
    
    # Compute time in milliseconds: (Sample * Ts) in seconds, then convert to ms
    data['time'] = pd.to_timedelta(data['Sample'] * Ts, unit='s')

    # Ensure ECG lead data are numeric
    data['MLII'] = pd.to_numeric(data['MLII'] , errors='coerce')
    data['V5'] = pd.to_numeric(data['V5'] ,errors='coerce')

    # Determine initial 4-second window based on the minimum time value in the data.
    start_time = data['time'].min()
    if pd.isnull(start_time):
        start_time = pd.Timedelta(seconds=0)
    end_time = start_time + pd.Timedelta(seconds=4)

    # Create Plotly figure
    fig = go.Figure()

    
    
    # Plot V5 lead in green
    fig.add_trace(go.Scatter(x=data['time'], y=data['V5'], 
                             mode='lines', name='V5', 
                             line=dict(color='green')))
    
    # Plot MLII lead in red
    fig.add_trace(go.Scatter(x=data['time'], y=data['MLII'], 
                             mode='lines', name='MLII', 
                             line=dict(color='red')))
    
    fig.update_layout(xaxis_title='Time (ms)', yaxis_title='ECG (mV)',
                      title="ECG Plot")
    fig.update_layout(
        xaxis=dict(
            title='Time (HH:MM:SS)',
            range=[start_time, end_time],
            fixedrange=True,
            rangeslider=dict(visible=True)
        ),
        yaxis=dict(title='ECG (mV)'),
        title="ECG Plot (4-second window)",
    )
    
    return fig


def diagnose_by_symptoms(symptoms, test_file):
    """
    Diagnose potential cardiac conditions based on symptom inputs and ECG findings,
    and return the top 3 probable diagnoses with their probability scores,
    recommended tests, and advice if testing is not available.

    Parameters:
    - symptoms (dict): Dictionary of symptoms with boolean or severity values.
      Expected keys: 'palpitations', 'chest_pain', 'dyspnea', 'syncope',
                     'fatigue', 'dizziness', 'edema'
    - test_file: Path to a file to obtain ECG findings (via predict_ecg_file)

    Returns:
    - result (list): List of top 3 diagnoses (dicts) with keys:
         'diagnosis': Name of the condition,
         'probability': Percentage probability,
         'recommended_tests': Recommended further tests,
         'advice_without_tests': Advice if tests are not available.
    """
    # Assume predict_ecg_file is defined elsewhere
    ecg_findings = predict_ecg_file(test_file)

    # Initialize scores for each potential diagnosis
    diagnosis_scores = {
        "Premature Atrial/Ventricular Beats": 0,
        "Paroxysmal Atrial Fibrillation (AF)": 0,
        "Supraventricular Tachycardia (SVT)": 0,
        "Coronary Artery Disease (Ischemia)": 0,
        "Acute Myocardial Infarction (AMI)": 0,
        "Pericarditis or Other Nonischemic Causes": 0,
        "Heart Failure / Cardiomyopathy": 0,
        "Ischemic Heart Disease": 0,
        "Pulmonary Causes (e.g., COPD, Pulmonary Embolism)": 0,
        "Arrhythmia-related Syncope": 0,
        "Bradyarrhythmia": 0,
        "Orthostatic Hypotension / Vasovagal Episodes": 0,
        "Non-Cardiac Causes (e.g., Anemia, Thyroid Dysfunction)": 0,
        "Arrhythmias (e.g., VT, Severe Bradycardia)": 0,
        "Conduction Abnormalities (e.g., LBBB, RBBB)": 0,
        "Medication Effects / Dehydration": 0,
        "Right-Sided Heart Failure": 0,
        "Renal Dysfunction": 0,
        "Venous Insufficiency": 0,
        "Underlying Structural Heart Disease (LBBB)": 0,
        "Possible Conduction Delay (RBBB)": 0,
        "High Atrial Ectopy - Increased AF Risk": 0,
        "High Ventricular Ectopy - Risk of PVC-induced Cardiomyopathy": 0,
    }

    # Mapping for test recommendations and advice for each diagnosis
    diagnosis_advice = {
        "Premature Atrial/Ventricular Beats": {
            "tests": "Holter monitor, standard ECG",
            "advice": "Often benign; monitor if symptoms worsen."
        },
        "Paroxysmal Atrial Fibrillation (AF)": {
            "tests": "Extended Holter monitoring or event recorder, Echocardiogram",
            "advice": "Evaluate atrial size and consider anticoagulation if AF is confirmed."
        },
        "Supraventricular Tachycardia (SVT)": {
            "tests": "ECG during episodes, electrophysiological study",
            "advice": "Medication or ablation may be needed if episodes are frequent."
        },
        "Coronary Artery Disease (Ischemia)": {
            "tests": "Stress test, Coronary CT or angiography",
            "advice": "Manage risk factors; initial treatment may be conservative if tests aren’t available."
        },
        "Acute Myocardial Infarction (AMI)": {
            "tests": "Cardiac enzymes, urgent ECG, coronary angiography",
            "advice": "Immediate treatment is necessary; if testing isn’t available, treat based on clinical suspicion."
        },
        "Pericarditis or Other Nonischemic Causes": {
            "tests": "Echocardiogram, inflammatory markers, ECG",
            "advice": "Often managed with anti-inflammatory medications if invasive testing isn’t possible."
        },
        "Heart Failure / Cardiomyopathy": {
            "tests": "Echocardiogram, BNP levels, Cardiac MRI",
            "advice": "Medical management and lifestyle modifications; imaging clarifies severity."
        },
        "Ischemic Heart Disease": {
            "tests": "Stress test, Echocardiogram, coronary angiography",
            "advice": "Risk factor management is essential; noninvasive tests can guide initial treatment."
        },
        "Pulmonary Causes (e.g., COPD, Pulmonary Embolism)": {
            "tests": "Chest X-ray, CT pulmonary angiography, pulmonary function tests",
            "advice": "Treat underlying lung disease; cardiac tests help if diagnosis remains unclear."
        },
        "Arrhythmia-related Syncope": {
            "tests": "Holter monitor, event recorder, tilt-table test",
            "advice": "Observation and medication adjustments may suffice if advanced tests aren’t available."
        },
        "Bradyarrhythmia": {
            "tests": "ECG, Holter monitor",
            "advice": "Consider pacemaker evaluation if symptoms are severe."
        },
        "Orthostatic Hypotension / Vasovagal Episodes": {
            "tests": "Blood pressure monitoring, tilt-table test",
            "advice": "Lifestyle changes and hydration often help."
        },
        "Non-Cardiac Causes (e.g., Anemia, Thyroid Dysfunction)": {
            "tests": "Complete blood count, thyroid function tests",
            "advice": "Medical management based on test results; symptoms may improve without invasive tests."
        },
        "Arrhythmias (e.g., VT, Severe Bradycardia)": {
            "tests": "ECG, Holter monitor, electrophysiological study",
            "advice": "Timely evaluation is crucial; if testing isn’t possible, manage risk factors and symptoms."
        },
        "Conduction Abnormalities (e.g., LBBB, RBBB)": {
            "tests": "ECG, Echocardiogram",
            "advice": "Often monitored over time; structural imaging can provide additional insights if needed."
        },
        "Medication Effects / Dehydration": {
            "tests": "Medication review, electrolyte panel",
            "advice": "Adjust medications and ensure proper hydration; invasive tests usually aren’t required."
        },
        "Right-Sided Heart Failure": {
            "tests": "Echocardiogram, BNP levels",
            "advice": "Diuretics and lifestyle changes are first-line; testing confirms the diagnosis."
        },
        "Renal Dysfunction": {
            "tests": "Renal function tests (BUN, Creatinine)",
            "advice": "Often managed conservatively unless severe."
        },
        "Venous Insufficiency": {
            "tests": "Doppler ultrasound",
            "advice": "Compression therapy and exercise are usually effective."
        },
        "Underlying Structural Heart Disease (LBBB)": {
            "tests": "Echocardiogram, Cardiac MRI",
            "advice": "These imaging tests assess structural damage; treatment depends on severity."
        },
        "Possible Conduction Delay (RBBB)": {
            "tests": "ECG, possibly Echocardiogram",
            "advice": "Often benign; further evaluation is warranted if symptoms persist."
        },
        "High Atrial Ectopy - Increased AF Risk": {
            "tests": "Holter monitor, event recorder",
            "advice": "Regular monitoring and risk factor modification are recommended."
        },
        "High Ventricular Ectopy - Risk of PVC-induced Cardiomyopathy": {
            "tests": "Holter monitor, Echocardiogram",
            "advice": "Evaluation of ventricular function is key; management can be conservative if tests are unavailable."
        }
    }
    
    # Score potential diagnoses based on symptoms
    # (Weights here are illustrative; adjust based on clinical guidelines)
    if symptoms.get('palpitations', False):
        diagnosis_scores["Premature Atrial/Ventricular Beats"] += 1.0
        diagnosis_scores["Paroxysmal Atrial Fibrillation (AF)"] += 0.8
        diagnosis_scores["Supraventricular Tachycardia (SVT)"] += 0.6

    if symptoms.get('chest_pain', False):
        diagnosis_scores["Coronary Artery Disease (Ischemia)"] += 1.0
        diagnosis_scores["Acute Myocardial Infarction (AMI)"] += 0.9
        diagnosis_scores["Pericarditis or Other Nonischemic Causes"] += 0.5

    if symptoms.get('dyspnea', False):
        diagnosis_scores["Heart Failure / Cardiomyopathy"] += 1.0
        diagnosis_scores["Ischemic Heart Disease"] += 0.8
        diagnosis_scores["Pulmonary Causes (e.g., COPD, Pulmonary Embolism)"] += 0.5

    if symptoms.get('syncope', False):
        diagnosis_scores["Arrhythmia-related Syncope"] += 1.0
        diagnosis_scores["Bradyarrhythmia"] += 0.8
        diagnosis_scores["Orthostatic Hypotension / Vasovagal Episodes"] += 0.5

    if symptoms.get('fatigue', False):
        diagnosis_scores["Heart Failure / Cardiomyopathy"] += 1.0
        diagnosis_scores["Ischemic Heart Disease"] += 0.8
        diagnosis_scores["Non-Cardiac Causes (e.g., Anemia, Thyroid Dysfunction)"] += 0.5

    if symptoms.get('dizziness', False):
        diagnosis_scores["Arrhythmias (e.g., VT, Severe Bradycardia)"] += 1.0
        diagnosis_scores["Conduction Abnormalities (e.g., LBBB, RBBB)"] += 0.8
        diagnosis_scores["Medication Effects / Dehydration"] += 0.5

    if symptoms.get('edema', False):
        diagnosis_scores["Right-Sided Heart Failure"] += 1.0
        diagnosis_scores["Renal Dysfunction"] += 0.8
        diagnosis_scores["Venous Insufficiency"] += 0.5

    # Incorporate ECG findings into scoring:
    if ecg_findings:
        if ecg_findings.get('LBBB', 0) > 0:
            diagnosis_scores["Underlying Structural Heart Disease (LBBB)"] += 1.0
        if ecg_findings.get('RBBB', 0) > 0:
            diagnosis_scores["Possible Conduction Delay (RBBB)"] += 0.8
        if ecg_findings.get('PACs', 0) > 30:
            diagnosis_scores["High Atrial Ectopy - Increased AF Risk"] += 1.0
        if ecg_findings.get('PVCs', 0) > 10:
            diagnosis_scores["High Ventricular Ectopy - Risk of PVC-induced Cardiomyopathy"] += 1.0

    # Calculate total score for normalization
    total_score = sum(diagnosis_scores.values())

    # Normalize scores to percentages if total_score > 0
    diagnosis_probabilities = {}
    if total_score > 0:
        for diag, score in diagnosis_scores.items():
            diagnosis_probabilities[diag] = (score / total_score) * 100
    else:
        # If no symptoms or ECG findings, assign zero probability
        for diag in diagnosis_scores.keys():
            diagnosis_probabilities[diag] = 0.0

    # Sort diagnoses by probability, descending
    sorted_diagnoses = sorted(diagnosis_probabilities.items(), key=lambda x: x[1], reverse=True)

    # Select the top 3 diagnoses
    top3 = sorted_diagnoses[:3]

    # Build the result list with diagnosis details
    result = []
    for diag, prob in top3:
        advice_info = diagnosis_advice.get(diag, {"tests": "N/A", "advice": "No advice available"})
        result.append({
            "diagnosis": diag,
            "probability": round(prob, 2),
            "recommended_tests": advice_info["tests"],
            "advice_without_tests": advice_info["advice"]
        })

    return result

