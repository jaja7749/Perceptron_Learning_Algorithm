# Perceptron Learning Algorithm

The Perceptron Learning Algorithm (PLA) is a supervised learning algorithm used for binary classification tasks. It was developed by Frank Rosenblatt in 1957 and is considered one of the simplest forms of artificial neural networks. The Perceptron is a single-layer neural network that can be used to classify input data into two classes.

Here's a step-by-step description of the Perceptron Learning Algorithm:

1. Initialization:
   - Initialize the weights ($`w`$) and bias ($`b`$) to small random values or zeros.
   - Assign a learning rate ($`\alpha`$), which determines the size of the steps taken during the learning process.

2. Input:
   - For each training example ($`x`$), augment it by adding a bias term ($`1`$) at the beginning of the input vector.

3. Activation Function:
   - Apply the activation function to the weighted sum of inputs. The commonly used activation function for perceptrons is the step function, which outputs 1 if the sum is greater than or equal to zero and 0 otherwise.
   - $`sign(x)=\left\{\begin{matrix} 1\ \ \mathrm{if}\ x\geq 0 \\ 0\ \ otherwise \end{matrix}\right.`$
  
4. Probability:
   - $`P(x) = sign(\omega_i ^{T}x_i+b)`$

5. Update Weights:
   - If the predicted output does not match the true output, update the weights and bias according to the Perceptron learning rule:
     - $`\omega _i^{(n+1)}=\omega _i^{(n)}+y^{i}x^{i},\ \mathrm{if}\ y^{i}(\omega ^{(n)T}x^{i})\leq 0`$
     - $`b=b+y^{i},\ \mathrm{if}\ y^{i}(\omega ^{(n)T}x^{i})\leq 0`$
   - Where:
     - $`\omega _i`$ is the weight associated with the $`i`$-th input.
     - $`x_i`$ is the $`i`$-th input feature.
     - $`y_i`$ is the true output.

6. Iteration:
   - Repeat steps 3 and 4 for each training example until the algorithm converges, meaning that the weights and bias no longer change significantly, or a predefined number of iterations is reached.

The Perceptron Learning Algorithm works well for linearly separable data, where it is possible to draw a straight line (or hyperplane in higher dimensions) to separate the two classes. However, it may not converge for data that is not linearly separable. In such cases, more advanced algorithms like the multilayer perceptron (MLP) or support vector machines (SVM) may be used.
