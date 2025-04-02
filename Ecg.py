#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm  #  for progress bar
import collections
import random
import glob
from sklearn.preprocessing import StandardScaler
from collections import Counter
from torch.utils.data import WeightedRandomSampler


# In[3]:


class ECGDataset(Dataset):
    def __init__(self, directory):
        self.csv_file_paths = [os.path.join(directory, fname) for fname in os.listdir(directory) if fname.endswith('.csv')]
        self.data = []
        self.labels = []

        #  Label mapping for classification
        label_mapping = {'N': 0, 'L': 1, 'R': 2, 'A': 3, 'V': 4}

        for file_path in self.csv_file_paths:
            file_name = os.path.basename(file_path).replace('.csv', '')  # Get ECG file name (e.g., "100")
            annotation_path = os.path.join(directory, f"{file_name}annotations.txt")  # Matching annotation file

            #  Load ECG Data
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.replace("'", "").str.strip()  # Fix column names
            feature_columns = df.columns[1:3]  # Select MLII & V5
            ecg_samples = df[feature_columns].values

            #  Load Annotations (Labels) with error handling
            try:
                with open(annotation_path, "r") as file:
                    lines = file.readlines()[1:]  #  Skip header row

                cleaned_data = []
                for line in lines:
                    parts = line.strip().split()  #  Split by whitespace
                    if len(parts) >= 3:  #  Ensure at least 3 columns exist
                        sample_idx, label = parts[1], parts[2]
                        label = "".join(filter(str.isalpha, label))  #  Extract only valid label letters
                        cleaned_data.append([int(sample_idx), label])

                #  Convert to DataFrame
                annotations = pd.DataFrame(cleaned_data, columns=["Sample", "Type"])

                #  Filter valid labels
                annotations = annotations[annotations["Type"].isin(label_mapping.keys())]
                annotations["Label"] = annotations["Type"].map(label_mapping)

            except Exception as e:
                print(f"🚨 Skipping {annotation_path} due to error: {e}")
                continue  #  Skip this file instead of crashing
            
            #  Match Labels to ECG Samples
            labels = np.zeros(len(df), dtype=np.int64)  # Default all labels to 'N' (0)
            for _, row in annotations.iterrows():
                sample_idx = int(row['Sample'])  # Ensure it's an integer index
                if 0 <= sample_idx < len(labels):  # Ensure index is valid
                    labels[sample_idx] = row['Label']  # Assign label

            self.data.append(ecg_samples)
            self.labels.append(labels)

        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

        #  Print label distribution before balancing
        label_counts = collections.Counter(self.labels.tolist())
        print("📌 Before Balancing - Label Counts:", label_counts)

        #  Balance dataset by undersampling class '0'
        zero_class_indices = [i for i, label in enumerate(self.labels) if label == 0]
        non_zero_indices = [i for i, label in enumerate(self.labels) if label != 0]

        #  Keep a maximum of 2x the non-zero labels for class '0'
        random.shuffle(zero_class_indices)
        zero_class_sample_size = min(len(zero_class_indices), len(non_zero_indices) * 2)
        zero_class_indices = zero_class_indices[:zero_class_sample_size]

        #  Combine balanced dataset
        selected_indices = zero_class_indices + non_zero_indices
        random.shuffle(selected_indices)

        #  Update dataset
        self.data = self.data[selected_indices]
        self.labels = self.labels[selected_indices]

        #  Print label distribution after balancing
        label_counts = Counter(self.labels.tolist())
        print("📌 After Balancing - Label Counts:", label_counts)

        #  **Normalize Data**
        print("📌 Applying StandardScaler normalization...")
        scaler = StandardScaler()
        self.data = scaler.fit_transform(self.data)  # Normalize features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float32)
        sample = sample.unsqueeze(0)  # ✅ Ensure shape (seq_length, input_size)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample, label


# In[4]:


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3, output_size=5):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #  Add dropout layer to prevent overfitting
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

        #  Add batch normalization for stable training
        self.batch_norm = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        #  Ensure input is 3D (batch_size, seq_length, input_size)
        if x.dim() == 2:  
            x = x.unsqueeze(1)  # Add sequence_length = 1 if missing

        #  Ensure hidden state batch size matches input batch size
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        #  Forward pass
        out, _ = self.lstm(x, (h0, c0))  # Pass hidden states
        out = self.batch_norm(out[:, -1, :])  # Apply batch normalization
        out = self.fc(out)  # Fully connected layer

        return out


# ## Understanding the LSTM Model for ECG Data
# 
# The `LSTMModel` class defines a Long Short-Term Memory (LSTM) neural network designed for processing ECG data. LSTMs are particularly effective for sequential data, making them suitable for analyzing heartbeat patterns over time.
# 
# ### Explanation of the Components
# 
# - **`input_size`**: Defines the number of input features per time step (e.g., number of ECG leads used).
# - **`hidden_size`**: Specifies the number of hidden units in each LSTM layer, controlling model complexity.
# - **`num_layers`**: Determines how many stacked LSTM layers are used for learning complex patterns.
# - **`output_size`**: Sets the dimension of the final output (e.g., number of classification categories).
# - **`lstm`**: The core sequential processing unit, which learns dependencies over time.
# - **`fc`**: A fully connected layer that maps the LSTM's last output to the desired prediction.
# 
# ### Explanation of the Forward Pass
# 
# - Initializes hidden and cell states (`h0`, `c0`) as zero tensors, ensuring proper state tracking.
# - Feeds the input sequence `x` through the LSTM layers.
# - Extracts only the last time step’s output (`out[:, -1, :]`), as it holds the learned representation for classification.
# - Applies the fully connected layer (`fc`) to generate final predictions.
# 
# ### Why This Matters
# 
# - LSTMs are well-suited for time-series and sequential analysis, making them effective for ECG classification.
# - The use of multiple LSTM layers helps capture long-term dependencies in heartbeat sequences.
# - Selecting only the last time step’s output ensures efficient prediction while reducing computational overhead.
# 
# ### Key Considerations
# 
# - Adjusting `hidden_size` and `num_layers` can significantly impact model performance.
# - Ensure input sequences have the correct shape (`batch_size, time_steps, input_size`).
# - Using GPU acceleration (`.to(device)`) improves training speed and efficiency.
# 
# This LSTM model serves as a powerful tool for ECG analysis, capable of learning complex sequential dependencies and making accurate predictions based on time-series heartbeat data.

# In[5]:


def evaluate(model, dataloader, device, return_logits=False):
    """
    Evaluates the model on the given dataloader.

    Works for both mid-training and post-training evaluation.

    Parameters:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for evaluation.
        device (torch.device): The device to run evaluation on.
        return_logits (bool): If True, returns logits along with loss and accuracy.

    Returns:
        tuple: (avg_loss, accuracy) or (avg_loss, accuracy, logits)
    """
    was_training = model.training  # Store if model was in training mode
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0
    all_logits = [] if return_logits else None  # Collect logits if needed

    with torch.no_grad():  # Disable gradients for validation
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to device
            
            outputs = model(inputs)  #  Fixed variable name (`inputs` instead of `sampels`)
            loss = F.cross_entropy(outputs, targets)  #  Compute loss correctly
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)  # Get predictions
            correct += (predicted == targets).sum().item()  #  Correct accuracy calculation
            total += targets.size(0)

            if return_logits:
                all_logits.append(outputs.cpu())

    avg_loss = total_loss / max(1, len(dataloader))  #  Avoid division by zero
    accuracy = correct / max(1, total)  #  Avoid division by zero

    model.train(was_training)  # Restore original training state

    if return_logits:
        return avg_loss, accuracy, torch.cat(all_logits, dim=0)  #  Return logits if needed
    return avg_loss, accuracy


# ## Understanding the Model Evaluation Process
# 
# Evaluating the model is essential for assessing its performance during and after training. The `evaluate` function helps compute loss and accuracy on a validation or test dataset to monitor model progress.
# 
# ### How the Evaluation Function Works
# 
# This function takes in a trained model, a dataset loader, and a device specification (`CPU` or `GPU`) to evaluate model performance.
# 
# ### key Considerations
# 
# - Ensure the evaluation dataset is separate from the training data.
# - Use a consistent batch size during evaluation for stable results.
# - If using a multi-class classification problem, adapt `F.cross_entropy` accordingly.
# 
# This evaluation function provides an efficient way to assess model performance, ensuring continuous monitoring and improvement throughout training.
# 
# 

# In[6]:


def save_checkpoint(epoch, model, batch_idx, optimizer, loss, accuracy, path="model/best_checkpoint.pth"):
    """
    Saves the model checkpoint only if:
    1. No previous model exists.
    2. New model has a higher accuracy than the previous one.
    
    Deletes inferior models automatically.
    """
    checkpoint_dir = os.path.dirname(path)

    # Ensure the directory exists
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"📁 Created directory: {checkpoint_dir}")

    # Check if a previous checkpoint exists
    if os.path.exists(path):
        try:
            prev_checkpoint = torch.load(path, map_location="cpu")  # Load safely
            prev_accuracy = prev_checkpoint.get('accuracy', 0)
        except Exception as e:
            print(f"⚠️ Error loading existing checkpoint: {e}")
            prev_accuracy = 0  # Assume no valid previous accuracy

        # Keep only if accuracy is better
        if accuracy <= prev_accuracy:
            print(f"🚫 New model NOT saved. Accuracy {accuracy:.4f} ≤ {prev_accuracy:.4f}")
            return  # Skip saving

        # Delete previous checkpoint (ensuring it's removed)
        try:
            os.remove(path)
            print(f"🗑️ Deleted inferior checkpoint (Accuracy: {prev_accuracy:.4f})")
        except Exception as e:
            print(f"⚠️ Failed to delete old checkpoint: {e}")

    #  Save the best model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'batch_idx': batch_idx,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }, path)

    print(f" New Best Checkpoint Saved at {path} (Accuracy: {accuracy:.4f})")


# ## Understanding Model Checkpointing
# 
# Saving and loading model checkpoints is crucial for maintaining training progress, preventing data loss, and resuming interrupted training efficiently. The `save_checkpoint` and `load_checkpoint` functions manage this process by storing and retrieving model parameters.
# 
# ### Saving Model Checkpoints
# 
# This function saves the model state along with training metadata, such as the current epoch, batch index, optimizer state, and performance metrics.
# 
# 
# 
# ### Loading Model Checkpoints
# 
# This function restores a saved model checkpoint, enabling training continuation without losing progress.
# 
# ### Why This Matters
# 
# - Prevents loss of progress due to system crashes or interruptions.
# - Allows for training continuation across different sessions or machines.
# - Enables model fine-tuning by starting from a pre-trained checkpoint.
# 
# ### Key Considerations
# 
# - Ensure the correct model and optimizer classes are passed when loading.
# - Store checkpoints periodically to avoid losing significant progress.
# - If resuming on a different machine, verify that the saved checkpoint is compatible with the new environment.
# 
# By implementing checkpointing, deep learning workflows become more efficient, minimizing training redundancy and ensuring reproducibility.
# 
# 

# In[7]:


def find_chackpoint(checkpoint_dir=r"C:\Users\shmue\projects\python\open_pojects\heart_ECG\model"):
    #  Find all checkpoint files matching "checkpoint_epoch_*.pth"
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth")))

    if not checkpoint_files:
        print(f"⚠️ No checkpoint found in '{checkpoint_dir}'. Starting from scratch.")
        return None, None, 0, 0, None, None  # Start from epoch 0

    #  Select the latest checkpoint (highest epoch number)
    latest_checkpoint = checkpoint_files[-1]  # Last file in sorted list
    return latest_checkpoint

def load_checkpoint(model_class, optimizer_class, device, checkpoint_dir="model/", model_args=None, optimizer_args=None):
    """
    Loads the latest model and optimizer checkpoint from the given directory.

    Parameters:
        model_class (torch.nn.Module): Model class to initialize the model.
        optimizer_class (torch.optim.Optimizer): Optimizer class to initialize the optimizer.
        device (torch.device): Device to load the model on.
        checkpoint_dir (str): Directory where checkpoints are stored.
        model_args (dict, optional): Dictionary of arguments for model initialization.
        optimizer_args (dict, optional): Dictionary of arguments for optimizer initialization.

    Returns:
        tuple: (model, optimizer, start_epoch, start_batch_idx, loss, accuracy)
    """
    latest_checkpoint = find_chackpoint()

    #  Load checkpoint
    checkpoint = torch.load(latest_checkpoint, map_location=device)

    #  Initialize model with given parameters
    model_args = model_args or {}
    model = model_class(**model_args).to(device)

    #  Initialize optimizer with given parameters
    optimizer_args = optimizer_args or {"lr": 0.001}
    optimizer = optimizer_class(model.parameters(), **optimizer_args)

    #  Load model and optimizer state
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint.get('epoch', 0)
    start_batch_idx = checkpoint.get('batch_idx', 0)
    loss = checkpoint.get('loss', None)
    accuracy = checkpoint.get('accuracy', None)

    print(f" Loaded checkpoint from {latest_checkpoint} (Epoch {start_epoch}, Batch {start_batch_idx}).")

    return model, optimizer, start_epoch, start_batch_idx, loss, accuracy


# In[ ]:


def train(train_loader, criterion, device, total_epochs=10):
    """
    Trains the model, resuming from the last checkpoint if available.
    
    Parameters:
        train_loader (torch.utils.data.DataLoader): Training data loader.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Training device (CPU/GPU).
        total_epochs (int): Number of epochs to train for.
    """
    model_args = {"input_size": 2, "hidden_size": 64, "num_layers": 2, "output_size": 5}
    

    optimizer_args = {"lr": 0.001}

    #  Load latest checkpoint or start fresh
    model, optimizer, start_epoch, start_batch_idx, loss, accuracy = load_checkpoint(
        model_class=LSTMModel, 
        optimizer_class=torch.optim.Adam, 
        device=device,
        checkpoint_dir="model/",
        model_args=model_args, 
        optimizer_args=optimizer_args
    )

    #  Handle missing checkpoint (Initialize a new model if needed)
    if model is None:
        model = LSTMModel(**model_args).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        start_epoch, start_batch_idx = 0, 0
        print("⚠️ No checkpoint found, starting training from scratch.")

    model.train()  #  Ensure model is in training mode
    criterion.to(device)  # Move loss function to the correct device

    for epoch in range(start_epoch, total_epochs):
        print(f"\n🚀 Started Epoch {epoch+1}/{total_epochs}")
        for batch_idx, (samples, labels) in enumerate(train_loader):
            #  Skip already processed batches in the current epoch
            if epoch == start_epoch and batch_idx < start_batch_idx:
                continue  

            samples, labels = samples.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(samples)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # eval
        avg_loss, accuracy = evaluate(model, train_loader, device, return_logits=False)
        
        #  Save checkpoint at the end of each epoch
        save_checkpoint(epoch, model, batch_idx, optimizer, loss, accuracy, path=f"model/checkpoint_epoch_{epoch}.pth")

        print(f"Epoch [{epoch+1}/{total_epochs}], Batch {batch_idx}/{len(train_loader)} - Val Loss = {avg_loss:.4f}, Val Accuracy = {accuracy:.4%}")

    print("\n🎉 Training complete!")


# ## Understanding the Model Training Process
# 
# Training a deep learning model involves iterating through the dataset, optimizing weights, and monitoring performance over multiple epochs. The `train` function facilitates this process while incorporating checkpointing for training resumption.
# 
# ### How the Training Function Works
# 
# This function takes a dataset loader, a loss function, and a device specification to train the model. It also supports saving and resuming training from a checkpoint.
# 
# ### Why This Matters
# 
# - Allows efficient training with automatic resumption.
# - Provides periodic evaluation to track learning progress.
# - Enables early stopping or hyperparameter tuning based on validation results.
# 
# ### Key Considerations
# 
# - Ensure batch skipping logic works correctly when resuming training.
# - Adjust learning rates dynamically if performance stagnates.
# - Regularly monitor validation accuracy to detect overfitting.
# 
# This training function ensures a robust workflow, combining efficient model training with checkpointing and evaluation to optimize ECG data analysis models.
# 
# 

# In[9]:


# Assuming paths and data are correct
# data_path = 'mitbih_database'
# dataset = ECGDataset(data_path)

# print(f'Dataset size: {dataset.__len__()}')

# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size

# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
# test_loader = DataLoader(test_dataset, batch_size=32)


# <h4> Model setup

# In[10]:


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# print('Runing on : cuda') if torch.cuda.is_available() else print('Runing on : cpu')

# criterion = nn.CrossEntropyLoss()
 
# train(train_loader, criterion, device, total_epochs=10)


# ## Understanding Data Loading for ECG Training
# 
# Properly loading and splitting a dataset is crucial for effective model training and evaluation. The following approach ensures that the ECG dataset is structured for training while maintaining a separate test set for validation.
# 
# ### How Data is Loaded and Split
# 
# - Loads the ECG dataset from the specified directory.
# - Ensures all `.csv` files in the directory are processed correctly.
# 
# * Splits the dataset into an **80% training set** and a **20% test set**.
# * Uses `torch.utils.data.random_split()` to randomly partition the data.
# 
# - Creates `DataLoader` objects for both the training and test datasets.
# - Enables **batch processing** with a batch size of 32.
# - `shuffle=True` is set for the training data to prevent model overfitting to specific data patterns.
# - The test dataset is not shuffled to maintain consistency in evaluation.
# 
# ### Why This Matters
# 
# - Ensures a proper training-test split for model evaluation.
# - Facilitates efficient data loading using PyTorch's `DataLoader`.
# - Randomization in the training set prevents the model from memorizing specific sequences.
# 
# ### Key Considerations
# 
# - The dataset split ratio can be adjusted based on dataset size and model requirements.
# - Larger batch sizes may improve training speed but require more memory.
# - Ensuring the test set remains separate is crucial for unbiased model evaluation.
# 
# This data loading process forms the foundation for training and evaluating deep learning models using ECG data, ensuring structured and reproducible experimentation.
# 
# 

# <h3> useage

# In[ ]:


def predict_ecg_file(file, window_size=200, stride=100):
    """
    Loads a single ECG .csv file and predicts label(s) using the trained model.

    Parameters:
        csv_file_path (str): Path to ECG .csv file.
        checkpoint_path (str): Path to model checkpoint.

    Returns:
        int: Predicted label for the file.
    """
    # path_csv = r'mitbih_database\{}.csv'.format(file_name)
    if file is None:
        raise ValueError("No file provided")
    try:
        # Read the CSV file directly from the uploaded file (BytesIO object)
        df = pd.read_csv(file)
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")
    df.columns = df.columns.str.replace("'", "").str.strip()
    # print(df.head())
    feature_columns = df.columns[1:3]  # Assume MLII and V5
    ecg_samples = df[feature_columns].values

    # print(ecg_samples)

    scaler = StandardScaler()
    ecg_samples = scaler.fit_transform(ecg_samples)
    
    # load devaice
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    latest_checkpoint = find_chackpoint()
    #  Load checkpoint
    checkpoint = torch.load(latest_checkpoint, map_location=device)

    #  Initialize model with given parameters
    model_args = {"input_size": 2, "hidden_size": 64, "num_layers": 2, "output_size": 5}
    
    model = LSTMModel(**model_args).to(device)

    #  Load model and optimizer state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    predictions = []

    for start in range(0, len(ecg_samples) - window_size + 1, stride):
        window = ecg_samples[start:start + window_size]
        input_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            predicted_label = torch.argmax(output, dim=1).item()
            predictions.append(predicted_label)

    if not predictions:
        return None

    prediction_df = pd.DataFrame(predictions, columns=['predictions'])
    # print(prediction_df)
    
    counts_dict = prediction_df.value_counts()
    
    LBBB = counts_dict.get(1, 0)
    RBBB = counts_dict.get(2, 0)
    APB = counts_dict.get(3, 0)
    PVC = (prediction_df['predictions'] == 4).mean() * 100

    # print(LBBB, RBBB, APB, PVC)
    
    return {
        "LBBB": LBBB,
        "RBBB": RBBB,
        "PACs": APB,
        "PVCs": PVC
    }


# need to improve model accurecy

# 
