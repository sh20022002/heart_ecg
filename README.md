# ECG Anomaly Detection and Diagnosis Pipeline

## 📌 Project Overview
This project aims to develop a **deep learning pipeline** for detecting anomalies in ECG data and diagnosing potential heart conditions. The system consists of two major components:

1. **ECG Signal-Based Model:** A deep learning model trained on MIT-BIH ECG signals to classify heart rhythms.
2. **ECG Image-Based Model:** A second model designed to process ECG images (uploaded photos) and detect anomalies.
3. **Pipeline System:** A fully integrated product that allows users to **upload ECG images** or capture photos, which will then be analyzed for potential cardiac abnormalities.

## 🚀 Key Features
- **Pre-trained Model for ECG Signals**: A trained **LSTM model** capable of classifying ECG signals.
- **ECG Image Processing**: Uses **image recognition techniques (CNN-based models)** to analyze ECG scans.
- **Seamless Pipeline**: Accepts either signal data or images, processes them, and returns a diagnosis.
- **Automatic Model Selection**: Determines whether the input is an ECG signal or an image and routes it accordingly.
- **Optimized Training and Model Selection**: Automatically **saves only the best model** while discarding inferior ones.

## 📂 Project Structure
```
├── data/                     # ECG signal and image datasets
├── models/                   # Saved models
│   ├── lstm_model.pth        # LSTM model for ECG signals
│   ├── cnn_model.pth         # CNN model for ECG images
├── src/
│   ├── dataset.py            # ECGDataset class (data preprocessing)
│   ├── train_signal.py       # Training script for ECG signal model
│   ├── train_image.py        # Training script for ECG image model
│   ├── inference.py          # Pipeline for ECG signal & image diagnosis
│   ├── utils.py              # Utility functions (e.g., normalization, augmentation)
├── notebooks/                # Jupyter notebooks for model exploration
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation (this file)
```

## 🔧 Setup & Installation
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-repo/ecg-diagnosis.git
   cd ecg-diagnosis
   ```

2. **Create a Virtual Environment & Install Dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate     # For Windows
   pip install -r requirements.txt
   ```

3. **Download & Prepare Data:**
   - Place **ECG signal CSVs** in `data/signals/`
   - Place **ECG images** in `data/images/`

## 🏋️ Training the Models
### **1. Train the ECG Signal Model (LSTM)**
```bash
python src/train_signal.py --epochs 50 --batch_size 32 --lr 0.001
```

### **2. Train the ECG Image Model (CNN)**
```bash
python src/train_image.py --epochs 50 --batch_size 32 --lr 0.001
```

## 📊 Model Evaluation
After training, evaluate models using:
```bash
python src/evaluate.py --model models/lstm_model.pth --data test_signals
python src/evaluate.py --model models/cnn_model.pth --data test_images
```

## 📸 ECG Image & Signal Inference
Run the pipeline for a **single image or signal file**:
```bash
python src/inference.py --input sample_ecg_image.png
python src/inference.py --input sample_ecg_signal.csv
```

## 🚀 Future Plans
- ✅ Deploy as a web API
- ✅ Implement **real-time ECG monitoring**
- ✅ Extend model capabilities for **multi-class classification**

## 📬 Contact & Contributions
If you have any ideas or would like to contribute, feel free to submit a pull request or contact me via **[your email/github]**.

