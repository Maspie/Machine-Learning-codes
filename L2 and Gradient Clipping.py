import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as nn_utils

# Define a simple linear network with L2 regularization applied via the optimizer
class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)

# Initialize the network
linear_net = LinearNet()

# Define an optimizer with L2 regularization (weight decay)
l2_optimizer = optim.SGD(linear_net.parameters(), lr=0.01, weight_decay=0.005)

# Define a more complex network for the gradient clipping example
class ComplexNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ComplexNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

# Set dimensions and initialize the complex network
batch_size, dim_in, dim_h, dim_out = 128, 2000, 200, 20
complex_model = ComplexNet(dim_in, dim_h, dim_out)

# Define the loss function and optimizer
loss_fn = nn.MSELoss(reduction='sum')
complex_optimizer = optim.Adam(complex_model.parameters(), lr=1e-4)

# Generate random data for input and output
input_X = torch.randn(batch_size, dim_in)
output_Y = torch.randn(batch_size, dim_out)

# Perform a single update step
complex_optimizer.zero_grad()
pred_y = complex_model(input_X)
loss = loss_fn(pred_y, output_Y)
loss.backward()

# Clip gradients and update model parameters
nn_utils.clip_grad_norm_(complex_model.parameters(), max_norm=5, norm_type=2)
complex_optimizer.step()

# Calculate the total norm of the gradients to verify clipping
total_norm = torch.sqrt(sum(p.grad.data.norm(2)**2 for p in complex_model.parameters()))
print(f"Total norm of gradients after clipping: {total_norm:.2f}")
