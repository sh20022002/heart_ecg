import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split



class ECGDataset(Dataset):
    def __init__(self, directory):
        self.file_paths = [os.path.join(directory, fname) for fname in os.listdir(directory) if fname.endswith('.csv')]
        self.data = []
        
        for file_path in self.file_paths:
            df = pd.read_csv(file_path, header=['sample', 'MLII', 'V5'])

            print(df.head())
            break
            
            self.data.append(df[['sample', 'MLII', 'V5']].values)

        self.data = np.concatenate(self.data, axis=0)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx, 1:], dtype=torch.float32)  # Use 'MLII' and 'V5'
        return sample



class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Load Data
data_path = 'mitbih_database'
dataset = ECGDataset(data_path)

# Split Data
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Hyperparameters
input_size = 2  # 'MLII' and 'V5'
hidden_size = 64
num_layers = 2
output_size = 1  # Assuming regression or binary classification task
num_epochs = 10
learning_rate = 0.001

# Model, Loss Function, Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()  # Change to appropriate loss function if necessary
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    for samples in train_loader:
        samples = samples.to(device).unsqueeze(-1)  # Adding a dummy dimension for LSTM

        # Forward pass
        outputs = model(samples)
        loss = criterion(outputs, samples[:, -1])  # Assuming last value is the target
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the Model
model.eval()
with torch.no_grad():
    total_loss = 0
    for samples in test_loader:
        samples = samples.to(device).unsqueeze(-1)

        outputs = model(samples)
        loss = criterion(outputs, samples[:, -1])
        total_loss += loss.item()
    
    print(f'Test Loss: {total_loss / len(test_loader):.4f}')

# Save the Model
torch.save(model.state_dict(), 'ecg_lstm_model.pth')

# Load the Model for Future Use
model.load_state_dict(torch.load('ecg_lstm_model.pth'))
model.eval()
