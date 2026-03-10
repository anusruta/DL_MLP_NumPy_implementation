# DL_MLP_NumPy_implementation
Building a multi-layer perceptron (MLP) neural network from scratch using only NumPy (no deep learning frameworks) to classify the Iris flower species.  Demonstrates neural networks, including forward propagation, backpropagation, gradient descent, and activation functions.

the Iris dataset was imported using the load_iris() function from the Scikit-learn library. The dataset features and corresponding target labels were separated into variables x and y. To make data exploration easier, a Pandas DataFrame was created using the feature values, and an additional column was added to represent the species names. Basic information about the dataset, such as its shape, sample records, and overall structure, was printed to better understand the data.

After loading the data, exploratory data analysis (EDA) was performed. Histograms were plotted for each feature to observe how the values are distributed. In addition, a pair plot was generated using Seaborn to visualize the relationships between different features and to see how the three iris species differ based on these measurements.

The dataset was then divided into training, validation, and test sets. First, 70% of the data was used for training and the remaining 30% was kept temporarily. This remaining portion was further split equally to create 15% validation data and 15% test data. Stratified sampling was used so that each split maintained a balanced representation of all three species.

Next, the feature values were standardized using the StandardScaler, which scales the data so that each feature has a mean close to zero and a standard deviation of one. This helps ensure that all features contribute equally during model training.

Finally, the target labels were converted into one-hot encoded vectors using the OneHotEncoder. This process transforms each class label into a binary vector, which allows the neural network to properly handle multi-class classification.



a multi-layer perceptron (MLP) neural network was built from scratch using NumPy. The network architecture was designed with four input neurons corresponding to the four features of the Iris dataset. Two hidden layers were used to learn patterns in the data: the first hidden layer contained 8 neurons, and the second hidden layer contained 6 neurons, both using the sigmoid activation function. The output layer consisted of 3 neurons, representing the three iris species, and used the softmax activation function to produce probability values for each class.

Several core functions were implemented to make the neural network work. First, the initialize_parameters() function was created to assign initial random values to the weights and biases with proper scaling. The sigmoid() function and its derivative were implemented for the hidden layers, while the softmax() function was used in the output layer to convert outputs into class probabilities. The forward_propagation() function calculated activations layer by layer to generate predictions. The compute_loss() function calculated the cross-entropy loss, which measures how well the model’s predictions match the true labels.

To train the model, backward_propagation() was implemented to calculate gradients of the loss with respect to the weights and biases using the chain rule of calculus. These gradients were then used in the update_parameters() function to adjust the weights using the gradient descent optimization method.

Finally, a training loop was created to train the network. The model was trained using mini-batch gradient descent with a batch size of 16 for up to 1000 epochs with a learning rate of 0.01. During training, the training loss and validation accuracy were monitored every 50 epochs to track the model’s performance. An early stopping mechanism was also included to stop training when the validation accuracy stopped improving, which helps prevent overfitting.



Part 3: Evaluation and Analysis – Summary

In this stage, the trained neural network was evaluated using the test dataset to measure how well the model performs on unseen data. Predictions were generated for the test samples and the overall test accuracy was calculated.

To analyze the performance in more detail, a confusion matrix was created to show how many samples from each class were correctly or incorrectly classified. In addition, a classification report was generated, which provided important evaluation metrics such as precision, recall, and F1-score for each iris species.

For visualization, Principal Component Analysis (PCA) was applied to reduce the feature space from four dimensions to two dimensions. This allowed the decision boundaries of the trained neural network to be visualized on a 2D plot, making it easier to observe how the model separates the different classes.

The performance of the custom neural network was also compared with Scikit-learn’s MLPClassifier to evaluate how closely the from-scratch implementation performs relative to a well-optimized library implementation.

Finally, after training and evaluation, the trained MLP model parameters and the PCA transformation model were saved. Saving these parameters allows the model to be reused later for predictions without retraining, and ensures that the same PCA transformation can be applied consistently when visualizing or processing new data.