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

# Perceptron Learning Algorithm on Different Distribution Data
we create the dataset from sklearn:
```ruby
from sklearn.datasets import make_blobs, make_circles, make_classification, make_moons, make_gaussian_quantiles

samples = 200
datasets = [
    make_blobs(n_samples=samples, centers=2, n_features=2, random_state=1),
    make_blobs(n_samples=samples, centers=2, n_features=2, random_state=6),
    make_moons(n_samples=samples, noise=0.15, random_state=0),
    make_circles(n_samples=samples, noise=0.15, factor=0.3, random_state=0),
    make_gaussian_quantiles(n_samples=samples, n_features=2, n_classes=2, random_state=0),
    make_classification(n_samples=samples, n_features=2, random_state=1, n_redundant=0, n_informative=1, n_clusters_per_class=1),
    make_classification(n_samples=samples, n_features=2, random_state=1, n_redundant=0, n_informative=2, n_clusters_per_class=1),
    make_classification(n_samples=samples, n_features=2, random_state=1, n_redundant=0, n_informative=2),
]
```
<img src="https://github.com/jaja7749/Perceptron_Learning_Algorithm/blob/main/images/different%20distribution%201.png" width="720">

First of all, we build PLA class to input $`X`$ and $`y`$ (label):
```ruby
class PLA:
    def __init__(self, X, label, random=None):
        self.X = X
        self.label = label
        np.random.seed(random)
        w = np.random.random(2)
        self.w = w
```
Then, we def sign function:
```ruby
    def sign(self, x):
        if x >= 0:
            return 1
        else:
            return -1 # or 0
```
Now, we can train our model:
```ruby
    def train(self, epoch=100, learning_rate = 0.1):
        for iteration in range(epoch):
            for i in range(len(self.X)):
                if (self.label[i]*(np.dot(self.w.T, self.X[i]))) <= 0:
                    self.w += learning_rate*(self.label[i]*self.X[i])
```
Finally, let's check our results:
<img src="https://github.com/jaja7749/Perceptron_Learning_Algorithm/blob/main/images/PLA%20result.png" width="720">

# Summary
The PLA can apply on the data is separable because PLA only have one perceptron. One perceptron can solve linear problem. However if the data is not separable, the model can not classified it. To solve the non_linear problem, people use more perceptrons to classified the data. That is as everyone know "Neural Network" now.
