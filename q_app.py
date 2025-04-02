import streamlit as st
from ECG_Prediction import diagnose_by_symptoms, image_plotly

# get the ecg file
file = st.file_uploader('chose the ecg(.csv) file', type=["csv"])


st.title("Cardiac Diagnosis by Symptoms & ECG Findings")
palpitations = st.sidebar.checkbox("Do you experience palpitations (a rapid, fluttering, or pounding heartbeat)?")
chest_pain = st.sidebar.checkbox("Do you have chest pain (discomfort, pressure, or tightness in your chest)?")
dyspnea = st.sidebar.checkbox("Do you suffer from dyspnea (shortness of breath or difficulty breathing)?")
syncope = st.sidebar.checkbox("Have you experienced syncope or presyncope (fainting or near-fainting episodes)?")
fatigue = st.sidebar.checkbox("Do you feel persistent fatigue or have exercise intolerance (unusual tiredness during activities)?")
dizziness = st.sidebar.checkbox("Do you often feel dizzy or lightheaded (a sensation of spinning or faintness)?")
edema = st.sidebar.checkbox("Do you notice peripheral edema (swelling in your legs or ankles)?")



# Build dictionaries from inputs
symptoms = {
    "palpitations": palpitations,
    "chest_pain": chest_pain,
    "dyspnea": dyspnea,
    "syncope": syncope,
    "fatigue": fatigue,
    "dizziness": dizziness,
    "edema": edema
}




if file is None:
    st.warning("Please upload a CSV file.")
else:
    if st.button("Diagnose"):
        st.header("ECG Findings..")
        diagnosis_results = diagnose_by_symptoms(symptoms, file)
        # Call the diagnosis function; file is the test file input    
        st.header("Diagnostic Recommendations")
        # Loop through the top 3 diagnosis results and display details
        for result in diagnosis_results:
            st.subheader(f"Diagnosis: {result['diagnosis']}")
            st.write(f"**Probability:** {result['probability']}%")
            st.write(f"**Recommended Tests:** {result['recommended_tests']}")
            st.write(f"**Advice (if tests unavailable):** {result['advice_without_tests']}")
            st.markdown("---")
        st.plotly_chart(image_plotly(file))