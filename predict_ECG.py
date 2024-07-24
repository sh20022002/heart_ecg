import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import torch.optim as optim

class ECGDataset(Dataset):
    def __init__(self, directory):
        self.csv_file_paths = [os.path.join(directory, fname) for fname in os.listdir(directory) if fname.endswith('.csv')]
        self.data = []
        
        dcolmns = ['sample', 'MLII', 'V5']
        for file_path in self.csv_file_paths:
            df = pd.read_csv(file_path,header=0, names=dcolmns)  # Assuming the CSV has no header by default; adjust if needed
            self.data.append(df[['sample', 'MLII', 'V5']].values)

        self.data = np.concatenate(self.data, axis=0)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx, 1:], dtype=torch.float32)  # Use 'MLII' and 'V5'
        label = torch.tensor(self.data[idx, 0], dtype=torch.float32)  # Assuming 'sample' column as label
        return sample, label

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        num_layers = 2
        hidden_size = 64
        h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Select last time step
        return out

# Assuming paths and data are correct
data_path = 'mitbih_database'
dataset = ECGDataset(data_path)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(input_size=2, hidden_size=64, num_layers=2, output_size=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    for samples, labels in train_loader:
        samples, labels = samples.to(device), labels.to(device).unsqueeze(1)  # Adjust dimensions
        optimizer.zero_grad()
        outputs = model(samples.unsqueeze(3))  # Adjust input dimensions
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')

# Model evaluation
model.eval()
total_loss = 0
with torch.no_grad():
    for samples, labels in test_loader:
        samples, labels = samples.to(device), labels.to(device).unsqueeze(1)
        outputs = model(samples.unsqueeze(1))
        loss = criterion(outputs, labels)
        total_loss += loss.item()
    print(f'Test Loss: {total_loss / len(test_loader):.4f}')
