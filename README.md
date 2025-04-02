# Heart ECG Anomaly Detection Pipeline

## Overview
The **Heart ECG Anomaly Detection Pipeline** is a project aimed at detecting anomalies in ECG signals and images. The system is designed to process ECG signal data, train machine learning models for anomaly detection, and ultimately extend its capabilities to analyze ECG images. The end goal is to create a pipeline that allows users to upload an ECG image and receive a diagnostic anomaly report.

## Features
- **ECG Signal Processing**: Uses the MIT-BIH dataset for ECG 2 lead signal classification.
- **Deep Learning Model**: Implements an LSTM-based neural network for anomaly detection.
- **Image-based ECG Analysis (Future Work)**: The next phase will involve training a computer vision model to analyze ECG images.
- **Pipeline Integration**: The final product will support image uploads and provide an anomaly diagnostic report.

## Repository Structure
```
heart-ecg/
│── .gitignore
│── README.md
│── requirements.txt
│── ECG_Data_head.txt
│── heart_conditions_category.txt
│── heart_terms.txt
│── Ecg.ipynb         # used for traning
|--- Ecg.py           # the file used as api for model prediction
│── ECG_Prediction.py # plots and diagnosis
|-- q_app.py          # streamlit app for qustiner loads csv ecg resoult and shows diagnosis
│── mitbih_database/  # ECG data (CSV files & annotations)
│── model/            # Saved model checkpoints
│── ecg/              # Virtual environment
│── __pycache__/
│── .vscode/          # VS Code configurations
```
![loads csv of ecg resoults and qustioner](screenshots\Screenshot_175351.png)

![diagnosis](screenshots\Screenshot_175902.png)

![diagnosis](screenshots\Screenshot_175918.png)

![plot ecg](screenshots\Screenshot_180018.png)

## Dataset
The dataset used for training is sourced from the **MIT-BIH Arrhythmia Database**. It consists of:
- **ECG Signals**: Contained in CSV files within the `mitbih_database/` directory.
- **Annotations**: Labels for each sample provided in corresponding `.txt` annotation files.

## Installation
To set up the environment, follow these steps:

1. **Clone the Repository**
   ```sh
   git clone https://github.com/yourusername/heart-ecg.git
   cd heart-ecg
   ```

2. **Create and Activate a Virtual Environment**
   ```sh
   python -m venv ecg
   source ecg/bin/activate  # On MacOS/Linux
   ecg\Scripts\activate     # On Windows
   ```

3. **Install Dependencies**
   ```sh
   pip install -r requirements.txt
   ```

## Model Training
To train the ECG classification model, run:
```sh
python ECG_Prediction.py
```
This script loads the dataset, trains the LSTM-based neural network, and saves the best-performing model.

## Model Checkpointing
The training script automatically saves the best-performing model checkpoint to the `model/` directory. The save function ensures that:
- A new model is saved only if it achieves a higher accuracy than existing checkpoints.
- Inferior models are deleted to optimize storage.

## Future Work
- **Image Processing Integration**: Implementing a CNN-based model for ECG image analysis.
- **Web API**: Developing an API for users to upload ECG images for automated diagnosis.
- **User Interface**: Creating a frontend to interact with the anomaly detection system.

## Contributing
If you'd like to contribute to this project, feel free to submit issues or pull requests on [GitHub](https://github.com/yourusername/heart-ecg).

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
🚀 **Heart ECG Anomaly Detection Pipeline** – Advancing ECG anomaly detection with deep learning!

