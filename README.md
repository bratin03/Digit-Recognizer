# Digit-Recognizer
Machine Learning

This Digit-Recognizer uses training data of hand written digits to train the model and then predicts the input handwritten digit.  
# Implementation:
## Test & Validation Set:
Training Dataset: The sample of data used to fit the model.  

Validation Dataset: The sample of data used to provide an unbiased evaluation of a model fit on the training dataset while tuning model hyperparameters. The evaluation becomes more biased as skill on the validation dataset is incorporated into the model configuration.
```bash
data = np.array(data)
np.random.shuffle(data)
val_rate = 0.2
val_num = int(data.shape[0] * val_rate)

m, n=data.shape

x_val = data[:val_num, 1:]
t_val = data[:val_num, 0]
x_train = data[val_num: , 1:]
t_train = data[val_num: , 0]

```
Here we divide the given dataset into a test and validation set in 80:20 ratio.

## Model:

![nn](https://user-images.githubusercontent.com/109758341/187070178-fa383b99-41e5-4c90-8c10-2de23d280eee.png)
Input: 784 pixel values of the image  
Hidden Layer: Consists of 10 units  
Output: Prediction Value corresponding to each digit 0-9  

## Initialization:
```bash
def init_params():
    W1 = np.random.randn(784, 10) *0.01
    b1 = np.zeros((1,10))
    W2 = np.random.randn(10, 10) *0.01
    b2 = np.zeros((1,10))
    
    return W1, b1, W2, b2
```
b1,b2: Zero Initialization  
W1,W2: Random Initialization(This serves the process of symmetry-breaking and gives much better accuracy. In this method, the weights are initialized very close to zero, but randomly.)

## Functions:
### ReLU:
```bash
def relu(x):
    return np.maximum(0, x)
```
![RELU](https://user-images.githubusercontent.com/109758341/187070794-0942d1c3-4d04-405d-858d-a3838fa6a89f.png)
### Softmax:
```bash
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 
    x = x - np.max(x) 
    return np.exp(x) / np.sum(np.exp(x))
```
![softmax](https://user-images.githubusercontent.com/109758341/187070938-b4547d0a-29f2-4605-92bd-96d68253f77d.png)
### Num_key:
```bash
def num_key(x):
    x.reshape(1, x.size)
    batch_size = len(x)
    t = np.zeros((batch_size, 10))
    t[np.arange(batch_size), x] = 1
    
    return t    
```
Returns a matrix for the training set where for each key number in 0-9 the matrix respective contains 1 if the number is that key and otherwise 0 
## Training
### Forward Propagation:
```bash
def forward(self, x, W1, b1, W2, b2):
        self.x = x
        self.W2 = W2
        self.A1 = np.dot(self.x, W1) + b1
        self.Z1 = relu(self.A1)
        self.A2 = np.dot(self.Z1, W2) + b2
        self.Y = softmax(self.A2)
```
For the hidden layer ReLU activation function is used  
For the final layer softmax activation function is used
### Backward Propagation:
```bash
    def backward(self, t):
        self.T = num_key(t)
        m = self.T.size
        dA2 = (self.Y - self.T) / m
        dW2 = np.dot(self.Z1.T, dA2)
        db2 = np.sum(dA2, axis = 0)    
        dZ1 = np.dot(dA2, self.W2.T)
        dA1 = dZ1 * relu(self.Z1) 
        db1 = np.sum(dA1, axis = 0) 
        dW1 = np.dot(self.x.T, dA1)
  ```
  The derivatives are calculated
  ### Updating:
  ```bash
  ef update(W1, b1, W2, b2, dW1, dW2, db1, db2, learning_rate):
    db1.reshape(1, db1.size)
    db2.reshape(1, db2.size)
    lr = learning_rate
    W1 = W1 - lr * dW1
    W2 = W2 - lr * dW2
    b1 = b1 - lr * db1
    b2 = b2 - lr * db2
    
    return W1, b1, W2, b2
```
The parameters are updated
### Training the parameters:
```bash
def train_network(x, t, iter, learning_rate):
    W1, b1, W2, b2 = init_params()
    prop = propagation()
    
    for i in range(iter):
        Y = prop.forward(x, W1, b1, W2, b2)
        dW1, dW2, db1, db2 = prop.backward(t)
        W1, b1, W2, b2 = update(W1, b1, W2, b2, dW1, dW2, db1, db2, learning_rate)
        if (i%10 == 0):
            print('iteration: ', i)
            print('accuracy: ', accuracy(Y, t))
 ```
 ## Predicting:
 ```bash
 def prediction(Y):
    return np.argmax(Y, axis = 1)
```
For a new input number, predicting it
 ## Accuracy:
 ```bash
 def accuracy(Y, t):
    K = prediction(Y)
    t.reshape(1, t.size)
    return np.sum(K == t) / K.size
```
Measuring the accuracy of the model

