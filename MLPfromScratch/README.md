# Multilayer Perceptron with Two Hidden Layers: Implementation from Scratch

This repository contains a simple Python implementation of a **Multilayer Perceptron (MLP)** neural network with two hidden layers. It includes:

1. **MLP class** (`mlp.py`), which defines the network structure, forward propagation, backpropagation, and basic training loop.  
2. **Example usage** demonstrating how to instantiate, train, and evaluate the MLP on a dataset such as the [MNIST digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).

## Contents

- **`mlp.py`**: Contains the `MLP` class, which:
  - Takes input size (`d_input`), two hidden-layer sizes (`d_hidden1`, `d_hidden2`), output size (`d_output`), and learning rate (`lr`).
  - Implements forward propagation with two hidden layers.
  - Implements backpropagation to update parameters via gradient descent.
  - Provides methods to train, predict, and compute accuracy.

- **`main.py`** (or similar script):
  - Demonstrates how to load a dataset (e.g., MNIST-like digits from `sklearn.datasets.load_digits`).
  - Instantiates the `MLP` with desired layer sizes and learning rate.
  - Trains the model on the training set.
  - Evaluates the model’s performance on a validation set.

- **Possible GUI app** (`DigitDrawApp` or similar):
  - (Optional) A Tkinter-based GUI that demonstrates drawing a digit and letting the MLP predict it.  
  - Captures user-drawn content and processes it to feed into the MLP.

## Getting Started

### Prerequisites

- **Python 3.7+**  
- Common Python libraries, such as:
  - `numpy`
  - `scikit-learn` (for dataset loading and splitting)
  - `matplotlib` (for data visualization)
  - `tkinter` (for the GUI demonstration)
  - `PIL` / `Pillow` (for image processing in the GUI)

Install these packages via:

```bash
pip install numpy scikit-learn matplotlib pillow
```

### Usage

1. **Train and Evaluate**  
   In the `main.py` (or whichever script you are using), you will see:

   ```python
   import numpy as np
   from sklearn.model_selection import train_test_split
   from sklearn.datasets import load_digits

   from mlp import MLP

   # 1. Load digits data
   digits = load_digits()
   X = digits.images.reshape((len(digits.images), -1))  # Flatten each image
   y = digits.target

   # 2. Split into training and validation
   X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

   # 3. Instantiate MLP with desired architecture
   d_input = 64         # each digit image is 8x8, so 64 pixels
   d_hidden1 = 32       # first hidden layer size
   d_hidden2 = 16       # second hidden layer size
   d_output = 10        # digits 0 through 9
   learning_rate = 0.001

   mlp = MLP(d_input, d_hidden1, d_hidden2, d_output, learning_rate)

   # 4. Train the MLP
   mlp.train(X_train, y_train, num_epochs=300)

   # 5. Evaluate accuracy on the validation set
   val_accuracy = mlp.accuracy(X_val, y_val)
   print("Validation Accuracy:", val_accuracy)
   ```

2. **GUI (Optional)**  
   If you implement a Tkinter-based GUI (like `DigitDrawApp`), it will let you draw digits on a canvas and use the trained MLP to predict them:
   - Make sure your MLP is trained (or loaded) before calling the GUI.
   - The GUI will capture the drawn digit, preprocess it into a `(1, 64)` input vector, and then call `mlp.predict(...)`.

## How It Works

1. **Network Architecture**:  
   - **Input Layer**: Takes flattened image vectors (or any features) of dimension `d_input`.
   - **Two Hidden Layers**: Each uses a **sigmoid** activation.  
   - **Output Layer**: Outputs `d_output` logits, then applies a **softmax** to convert them into class probabilities.

2. **Training**:  
   - **Forward Pass**:
     1. $ z_1 = XW_1 + b_1 $, $ a_1 = \text{sigmoid}(z_1) $  
     2. $ z_2 = a_1W_2 + b_2 $, $ a_2 = \text{sigmoid}(z_2) $  
     3. $ z_3 = a_2W_3 + b_3 $, $ \text{probs} = \text{softmax}(z_3) $
   - **Loss**: Uses cross-entropy loss: $-\log(\text{probs}_{\text{true_class}})$.
   - **Backpropagation** updates all three sets of weights (`W1`, `W2`, `W3`) and biases (`b1`, `b2`, `b3`) via gradient descent.

3. **Prediction**:  
   - Performs the same forward pass on new data and picks the class with the highest probability (`np.argmax`).