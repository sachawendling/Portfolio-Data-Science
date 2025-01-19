import numpy as np

class MLP:

    def __init__(self, d_input, d_hidden1, d_hidden2, d_output, lr):

        """
        Initialize the Multilayer Perceptron (MLP) model with two hidden layers.

        Parameters:
        -----------
        d_input : int
            The dimensionality of the input data (number of features).
        d_hidden1 : int
            The number of neurons in the first hidden layer.
        d_hidden2 : int
            The number of neurons in the second hidden layer.
        d_output : int
            The number of output classes (dimensionality of the output layer).
        lr : float
            The learning rate for gradient descent optimization.

        Attributes:
        -----------
        W1 : numpy.ndarray
            Weight matrix for the first (input-to-hidden) layer, shape (d_input, d_hidden1).
        b1 : numpy.ndarray
            Bias vector for the first hidden layer, shape (1, d_hidden1).
        W2 : numpy.ndarray
            Weight matrix for the second (hidden-to-hidden) layer, shape (d_hidden1, d_hidden2).
        b2 : numpy.ndarray
            Bias vector for the second hidden layer, shape (1, d_hidden2).
        W3 : numpy.ndarray
            Weight matrix for the third (hidden-to-output) layer, shape (d_hidden2, d_output).
        b3 : numpy.ndarray
            Bias vector for the third layer, shape (1, d_output).

        Notes:
        ------
        - This method sets up an MLP with two hidden layers, along with
          the relevant weights, biases, and learning rate.
        - The weights are initialized with small random values (using a
          variance-scaling approach to help with stable forward passes).
        - The bias vectors are initialized to zeros for simplicity.
        """

        # Set up MLP dimensions
        self.d_input = d_input
        self.d_hidden1 = d_hidden1
        self.d_hidden2 = d_hidden2
        self.d_output = d_output

        # Set up learning rate
        self.lr = lr

        # Set up weights and biases
        # Layer 1 (input -> hidden1)
        self.W1 = np.random.randn(d_input, d_hidden1) * np.sqrt(2. / d_input) 
        self.b1 = np.zeros((1, d_hidden1))

        # Layer 2 (hidden1 -> hidden2)
        self.W2 = np.random.randn(d_hidden1, d_hidden2) * np.sqrt(2.0 / d_hidden1)
        self.b2 = np.zeros((1, d_hidden2))

        # Layer 3 (hidden2 -> output)
        self.W3 = np.random.randn(d_hidden2, d_output) * np.sqrt(2.0 / d_hidden2)
        self.b3 = np.zeros((1, d_output))

    def __forward_layer(self, X, W, b):
        
        """
        Compute the linear transformation for a single layer: X * W + b.

        Parameters:
        -----------
        X : numpy.ndarray
            Input matrix with shape (n_samples, n_features), where `n_samples` is the number of
            data points and `n_features` is the dimensionality of the input.
        W : numpy.ndarray
            Weight matrix with shape (n_features, n_neurons), where `n_neurons` is the number
            of neurons in the current layer.
        b : numpy.ndarray
            Bias vector with shape (1, n_neurons), where `n_neurons` matches the number of
            neurons in the current layer.

        Returns:
        --------
        numpy.ndarray
            The output of the linear transformation, with shape (n_samples, n_neurons).
        """

        return np.dot(X, W) + b
    
    def __sigmoid(self, x):

        """
        Compute the sigmoid activation function.

        Parameters:
        -----------
        x : numpy.ndarray
            Input array, which can be of any shape, containing the pre-activation values.

        Returns:
        --------
        numpy.ndarray
            Output array with the same shape as `x`, where each element is transformed
            by the sigmoid function: 1 / (1 + exp(-x)).
        """

        x_clipped = np.clip(x, -709, 709)  # 709 is near the limit for double precision
        return 1 / (1 + np.exp(-x_clipped))
    
    def __softmax(self, z):

        """
        Compute the softmax activation function.

        Parameters:
        -----------
        z : numpy.ndarray
            Input array with shape (n_samples, n_classes), where `n_samples` is the
            number of data points and `n_classes` is the number of output classes.

        Returns:
        --------
        numpy.ndarray
            Output array with the same shape as `z`, where each row represents a
            probability distribution over the classes. The values in each row sum to 1.
        """

        exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))  # shift for numerical stability
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def __forward(self, X):

        """
        Perform forward propagation through the MLP.

        Parameters:
        -----------
        X : numpy.ndarray
            Input data with shape (n_samples, d_input), where `n_samples` is the number
            of data points and `d_input` is the number of input features.

        Returns:
        --------
        tuple
            a1, a2, probs
            - a1 : Activation of the first hidden layer, shape (n_samples, d_hidden1).
            - a2 : Activation of the second hidden layer, shape (n_samples, d_hidden2).
            - probs : Probability distribution over classes, shape (n_samples, d_output).
        """

        # Layer 1
        z1 = self.__forward_layer(X, self.W1, self.b1)
        a1 = self.__sigmoid(z1)

        # Layer 2
        z2 = self.__forward_layer(a1, self.W2, self.b2)
        a2 = self.__sigmoid(z2)

        # Layer 3 (output)
        z3 = self.__forward_layer(a2, self.W3, self.b3)
        probs = self.__softmax(z3)

        return a1, a2, probs

    def train(self, data_X, data_y, num_epochs, print_loss=True):

        """
        Train the Multilayer Perceptron (MLP) using gradient descent.

        Parameters:
        -----------
        data_X : numpy.ndarray
            The input data for training, with shape (n_samples, d_input), where
            `n_samples` is the number of training samples and `d_input` is the number of features.
        data_y : numpy.ndarray
            The target labels for training, with shape (n_samples,), where each value represents
            the class index (0-based) for the corresponding input sample.
        num_epochs : int
            The number of training iterations (epochs) to perform.
        print_loss : bool, optional (default=True)
            Whether to print the loss value at regular intervals during training.

        Returns:
        --------
        None
            The function modifies the weights and biases of the MLP in place as
            it optimizes the model to minimize the cross-entropy loss.
        """

        n_samples = len(data_X)

        for i in range(num_epochs):

            # Forward propagation
            a1, a2, probs = self.__forward(data_X)

            # Estimate the cross entropy loss
            correct_logprobs = -np.log(probs[range(len(data_y)), data_y])
            data_loss = np.sum(correct_logprobs) / len(data_X)

            # Backpropagation

            # 1. Output layer gradient
            delta3 = probs
            delta3[np.arange(n_samples), data_y] -= 1  # subtract 1 for correct class
            dW3 = np.dot(a2.T, delta3)
            db3 = np.sum(delta3, axis=0, keepdims=True)

            # 2. Second hidden layer gradient
            delta2 = np.dot(delta3, self.W3.T) * a2 * (1 - a2)
            dW2 = np.dot(a1.T, delta2)
            db2 = np.sum(delta2, axis=0, keepdims=True)

            # 3. First hidden layer gradient
            delta1 = np.dot(delta2, self.W2.T) * a1 * (1 - a1)
            dW1 = np.dot(data_X.T, delta1)
            db1 = np.sum(delta1, axis=0, keepdims=True)

            # Gradient descent parameter update
            self.W3 -= self.lr * dW3
            self.b3 -= self.lr * db3
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

            # Loss display
            if print_loss and i % 10 == 0:
                print("Epoch:", i, "Loss:", data_loss)

    def predict(self, X):

        """
        Predict the class labels for given input data.

        Parameters:
        -----------
        X : numpy.ndarray
            Input data with shape (n_samples, d_input), where `n_samples` is the number
            of data points and `d_input` is the number of input features.

        Returns:
        --------
        numpy.ndarray
            Predicted class labels with shape (n_samples,), where each element is the
            index of the class with the highest probability for the corresponding input sample.
        """

        # Forward propagation
        _, _, probs = self.__forward(X)

        # Return the predicted class labels (index of the max probability in each row)
        return np.argmax(probs, axis=1)
    
    def accuracy(self, X, y_true):

        """
        Compute the accuracy of the model's predictions.

        Parameters:
        -----------
        X : numpy.ndarray
            Input data with shape (n_samples, d_input), where `n_samples` is the number
            of data points and `d_input` is the number of input features.
        y_true : numpy.ndarray
            True class labels with shape (n_samples,), where each element is the
            ground truth class index (0-based) for the corresponding input sample.

        Returns:
        --------
        float
            The accuracy of the model, calculated as the proportion of correctly
            predicted samples, ranging from 0 to 1.
        """

        y_pred = self.predict(X)
        return np.mean(y_true == y_pred)
