import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import torch.optim as optim
import numpy as np

df = pd.read_csv('./coin_Bitcoin.csv')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = df[['High', 'Low', 'Open']].values
y = df[['Close']].values

scaler_x = StandardScaler()
scaler_y = StandardScaler()
x = scaler_x.fit_transform(x)
y = scaler_y.fit_transform(y)

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
train_x = torch.tensor(train_x.astype(np.float32)).unsqueeze(1)
train_y = torch.tensor(train_y.astype(np.float32))
test_x = torch.tensor(test_x.astype(np.float32)).unsqueeze(1)

class BitCoinDataSet(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

hidden_size = 512
num_layers = 2
learning_rate = 1e-3
batch_size = 64
epoch_size = 10

train_dataset = BitCoinDataSet(train_x, train_y)
test_dataset = BitCoinDataSet(test_x, test_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

class RNN(nn.Module):
    def __init__(self, input_feature_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(input_feature_size, hidden_size, num_layers, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        x = self.relu(self.linear1(x[:, -1, :]))
        x = self.linear2(x)
        return x

input_feature_size = train_x.size(2)
rnn = RNN(input_feature_size, hidden_size, num_layers).to(device)
criteria = nn.MSELoss()
optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)

rnn.train()
for epoch in range(epoch_size):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = rnn(inputs)
        loss = criteria(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 100 == 99:
            print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0
print('Finished Training')

rnn.eval()
prediction, ground_truth = [], []
with torch.no_grad():
    for data in test_loader:
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = rnn(inputs)
        prediction.extend(outputs.detach().cpu().numpy())
        ground_truth.extend(targets.cpu().numpy())

prediction = np.array(prediction).reshape(-1, 1)
ground_truth = np.array(ground_truth).reshape(-1, 1)
prediction = scaler_y.inverse_transform(prediction)
ground_truth = scaler_y.inverse_transform(ground_truth)
r2score = r2_score(ground_truth, prediction)
print('R^2 Score:', r2score)
