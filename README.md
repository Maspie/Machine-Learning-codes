## Machine Learning Codes

Different Machine Learning codes-


### Generic K-Means
- Load and Prepare Data:
Import libraries and load dataset. Select Column1 and Column2 for clustering.

- Elbow Method:
Calculate WCSS for 1-10 clusters to find the optimal number. Plot WCSS to identify the elbow point.

-  Optimal Clustering:
Choose 5 clusters based on the elbow graph. Apply K-Means clustering to the data.

- Visualize Clusters:
Plot data points color-coded by cluster and highlight centroids in red.

This approach unveils hidden patterns and structures in your data efficiently.

### L2 and Gradient Clipping

- Key Features
L2 Regularization:

- Purpose: Prevents overfitting by penalizing large weights.
- Implementation: Uses weight_decay in the SGD optimizer.
- Gradient Clipping:

- Purpose: Prevents exploding gradients for stable training.
- Implementation: Uses nn_utils.clip_grad_norm_ to clip gradients.
- Process
Define Networks: Simple linear and complex networks.
Apply L2 Regularization: Use weight_decay in the optimizer.
Clip Gradients: Use nn_utils.clip_grad_norm_ during backpropagation.
Train Models: Forward pass, compute loss, backpropagate, and update parameters.
Verify Clipping: Print total norm of gradients to confirm clipping.

### Recurrent

- Key Features
Data Preprocessing:

- Standardization: Scale features and target values using StandardScaler.
Train-Test Split: Split the data into training and testing sets.
Dataset and DataLoader:

- Custom Dataset: Define BitCoinDataSet class for handling Bitcoin data.
- DataLoader: Create data loaders for batch processing.
Model Architecture:

- RNN with LSTM: Define an RNN model using LSTM layers for sequential data processing.
Training:

- Loss and Optimizer: Use MSE loss and Adam optimizer.
- Training Loop: Train the model over multiple epochs with batch processing and loss tracking.
- Evaluation:

- Predictions: Make predictions on the test set.
- Performance Metric: Calculate R2 score to evaluate model performance.
Process
Load and Preprocess Data:

Read CSV data.
Standardize features and targets.
Split data into training and testing sets.
Convert data to PyTorch tensors.
- Define Dataset and DataLoader:

Implement BitCoinDataSet class.
Initialize DataLoader for training and testing data.
- Build and Train RNN Model:

Define RNN model architecture with LSTM layers.
Set loss function and optimizer.
Train the model, tracking loss over epochs.
- Evaluate Model:

Make predictions on the test set.
Inverse transform predictions to original scale.
Calculate and print the R2 score.

### Temperature forecast

- Load Dataset: Read CSV and create a time index.
- Prepare Data: Define feature (Time) and target (Temperature (C)).
- Train Model: Linear regression on Time and Temperature (C).
- Visualize: Plot original and predicted temperatures, and lagged temperature data.

### Action Recognition 1
a comprehensive pipeline for preprocessing and classifying video data using deep learning techniques. It begins with functions for extracting, resizing, and normalizing frames from a video to prepare datasets for training. A 3D convolutional neural network (3D CNN) is then designed and trained to classify videos into predefined categories based on their visual content. The model employs layers like Conv3D and MaxPooling3D for feature extraction and spatial downsampling, respectively, and uses dropout for regularization to prevent overfitting.

Key tools used include:

OpenCV (cv2): For video processing tasks like frame extraction and resizing.
NumPy: For efficient array operations.
TensorFlow/Keras: For building and training the neural network.
Matplotlib and Seaborn: For visualization, such as plotting the training history and confusion matrix.
Pickle: For saving and loading the model's training history and label encoder.
The complete process includes training the model with a dataset, evaluating its performance, and using it to predict the category of new videos, showcasing the effectiveness of 3D CNNs in understanding and categorizing temporal and spatial patterns in video data.


