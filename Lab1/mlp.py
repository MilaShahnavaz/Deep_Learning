import numpy as np
import data_generator

# Different activations functions


def activation(x, activation):

    # TODO: specify the different activation functions
    # 'activation' could be: 'linear', 'relu', 'sigmoid', or 'softmax'
    if activation == 'linear':
        return x
    elif activation == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    elif activation == 'relu':
        return np.maximum(0.0, x)
    elif activation == 'softmax':
        return (np.exp(x)/np.exp(x).sum())
        # TODO
    else:
        raise Exception("Activation function is not valid", activation)

# -------------------------------
# Our own implementation of an MLP
# -------------------------------


class MLP:
    def __init__(
        self,
        dataset,         # DataGenerator
    ):
        self.dataset = dataset

    # Set up the MLP from provided weights and biases
    def setup_model(
        self,
        W,                   # List of weight matrices
        b,                   # List of bias vectors
        activation='linear'  # Activation function of layers
    ):
        self.activation = activation

        # TODO: specify the number of hidden layers based on the length of the provided lists
        self.hidden_layers = len(W)-1

        self.W = W
        self.b = b

        # TODO: specify the total number of weights in the model (both weight matrices and bias vectors)
        self.N = sum(w.size + bi.size for w, bi in zip(W, b))

        print('Number of hidden layers: ', self.hidden_layers)
        print('Number of model weights: ', self.N)

    # Feed-forward through the MLP
    def feedforward(
        self,
        x      #The input data points, where each row represents one data point.
    ):
        # TODO: specify a matrix for storing output values for each input
        y = np.zeros((x.shape[0], self.dataset.K))

        # TODO: implement the feed-forward layer operations
        # 1. Specify a loop over all the datapoints
        # 2. Specify the input layer (2x1 matrix)
        # 3. For each hidden layer, perform the MLP operations
        #    - multiply weight matrix and output from previous layer
        #    - add bias vector
        #    - apply activation function
        # 4. Specify the final layer, with 'softmax' activation
        # Process each data point through the network
        for i in range(x.shape[0]):
            
            # Initialize the input to the first layer as the i-th data point
            layer_input = x[i, :][:, np.newaxis]

            # Process each hidden layer
            for layer in range(self.hidden_layers):
                # Calculate pre-activation values by multiplying weights with the input and adding the bias
                z = np.dot(self.W[layer], layer_input) + self.b[layer]
                # Apply the activation function to get the input for the next layer
                layer_input = activation(z, self.activation)

            # Calculate the final layer output with softmax activation
            final_z = np.dot(self.W[-1], layer_input) + self.b[-1] 
            softmax_output =  activation(final_z, 'softmax')
            y[i, :] = softmax_output[:, 0]

        return y

    # Measure performance of model
    def evaluate(self):
        print('Model performance:')

        # TODO: formulate the training loss and accuracy of the MLP
        # TODO: formulate the test loss and accuracy of the MLP
        # Assume the mean squared error loss
        # Hint: For calculating accuracy, use np.argmax to get predicted class
        
        # Feedforward to get predictions for training and testing data
        train_predictions = self.feedforward(self.dataset.x_train)
        test_predictions = self.feedforward(self.dataset.x_test)

        # Calculate the mean squared error for training data
        train_errors = (train_predictions - self.dataset.y_train_oh)**2
        train_loss = np.mean(train_errors)

        # Calculate the mean squared error for testing data
        test_errors = (test_predictions - self.dataset.y_test_oh)**2
        test_loss = np.mean(test_errors)

        train_acc = np.mean(np.argmax(train_predictions, 1) == self.dataset.y_train)
        test_acc = np.mean(np.argmax(test_predictions, 1) == self.dataset.y_test)

        print("\tTrain loss:     %0.4f" % train_loss)
        print("\tTrain accuracy: %0.2f" % train_acc)
        
        print("\tTest loss:      %0.4f" % test_loss)
        print("\tTest accuracy:  %0.2f" % test_acc)

