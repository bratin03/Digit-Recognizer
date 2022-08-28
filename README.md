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
