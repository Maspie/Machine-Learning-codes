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
