import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.datasets
import tkinter as tk

from mlp import MLP 
from app import DigitDrawApp

if __name__ == "__main__":

    # Load the MNIST dataset
    digits = sklearn.datasets.load_digits()
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1)) # We reshape the images into vector
    y = digits.target

    # Split the dataset into 80% training and 20% validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    # Initialize the model
    d_input = 64
    d_hidden1 = 64
    d_hidden2 = 32
    d_output = 10
    learning_rate = 0.001
    mlp = MLP(d_input, d_hidden1, d_hidden2, d_output, learning_rate)

    # Train the model on the training set 
    # and display the model accuracy according to the validation set
    mlp.train(X_train, y_train, num_epochs=1000)
    print("The accuracy obtained on the validation set is:", mlp.accuracy(X_val, y_val))

    root = tk.Tk()
    app = DigitDrawApp(root, model=mlp)
    root.title("Digit Prediction")
    root.mainloop()
