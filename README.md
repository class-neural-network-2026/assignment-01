# Logistic regression for a binary classification

## 1. Objective

- Logistric regression problem (gradient, accuracy)
- PyTorch library (Tensor, Dataset, DataLoader)
- Neural Network (matrix multiplication, activiation function)
- Stochastic Gradient Descent (mini-batch size, learning rate, number of epochs)

## 2. Baseline notebook code

- [assignment_01.ipynb](assignment_01.ipynb)

## 3. Utility codes

#### You have to complete the following codes:

- [MyModel.py](MyModel.py)
- [MyOptim.py](MyOptim.py)

#### You do not have to modify the following codes:

- [MyDataset.py](MyDataset.py)
- [MyEval.py](MyEval.py)
- [MyResult.py](MyResult.py)

## 4. Data

- mnist images of size $32 \times 32$ for digits 0 and 1
  - train ($500 \times 2$ classes)
  - test ($800 \times 2$ classes)

## 5. Linear function

- neural network $f_w(x)$ consists of a linear layer followed by the `sigmoid` activation function
- a linear layer consists of a linear transformation $\mathbb{R}^p \mapsto \mathbb{R}^q$
- a linear transformation can be obtained by a matrix multiplication:
> $$
> y = A x,
> $$
> where $A \in \mathbb{R}^{q \times p}$, $x \in \mathbb{R}^p$ and $y \in \mathbb{R}^q$

- neural network $f_w(x)$ for input $x$ is defined by:
> $$
> f_w(x) = \sigma( A x ),
> $$
> where $w$ denotes weights in the linear layer and $\sigma$ denotes sigmoid function defined by:
> $$
> \sigma(z) = \frac{1}{1 + \exp(-z)}
> $$

- output $h = f_w(x)$ of the neural network $f_w(x)$ for input $x$ is considered as prediction value for the class of input as follows:
> $$
> \begin{cases}
> l(x) = 0 & \colon h < 0.5 \\
> l(x) = 1 & \colon h \ge 0.5,
> \end{cases}
> $$
> where $l(x)$ denotes a label function that determines the class of $x$

## 6. Objective function

- objective function is defined by:
> $$
> \mathcal{L}(w) = \frac{1}{n} \sum_{i=1}^{n} \ell_i(w),
> $$
> where $\ell_i(w)$ denotes the loss for a pair of data $x_i$ and label $y_i$ as defined by:
> $$
> \ell_i(w) = - (y_i \log{(f_w(x_i))} + (1 - y_i) \log{(1 - f_w(x_i))})
> $$

## 7. Gradient

- gradient of the loss $\mathcal{L}(w)$ with respect to the weight $w$ is defined by:
> $$
> \nabla \mathcal{L}(w) = \frac{1}{n} \sum_{i=1}^{n} \nabla \ell_i(w),
> $$
> where the gradient of $\ell_i(w)$ for each pair of input $(x_i, y_i)$ is defined by:
> $$
> \begin{aligned}
> \nabla \ell_i(w) & = - y_i \frac{1}{f_w(x_i)} \frac{\partial f_w(x_i)}{\partial w} - (1 - y_i) \frac{1}{1 - f_w(x_i)} \frac{\partial (1 - f_w(x_i))}{\partial w}\\
> & = \left( f_w(x_i) - y_i \right) x_i
> \end{aligned}
> $$
> where we have:
> $$
> \frac{d \, \sigma(z)}{d \, z} = \sigma(z) (1 - \sigma(z))
> $$

## 8. Optimization by Stochastic Gradient Descent

- gradient descent step with a mini-batch is given as follows:
> $$
> \begin{aligned}
> w^{(t+1)} & = w^{(t)} - \eta \nabla \mathcal{L}(w)\\
> & = w^{(t)} - \eta \frac{1}{n} \sum_{i=1}^{n} \nabla \ell_i(w)
> \end{aligned}
> $$
> where $\eta$ denotes the learning rate and $n$ denotes the number of training data

## 9. Configuration

- batch size
- learning rate
- number of epochs
- initialization of model parameters
- number of layers
- activation functions
- dimension of the linear maps

## 10. GitHub history

- `commit` should be made at least 30 times
- the message of `commit` should be informative on the working progress
- the history of `commit` should effectively indicate the coding progress


## 11. Evaluation

- average of the training accuracy and the testing accuracy
- the average value will be your score
- the test will be performed based on the Autograding plateform at GitHub Classroom
- the mock test will be performed in the evening of the day before the next class date
- questions related to the assignment are recommended to be asked during the next class

## 12. Submission at GitHub Classroom

- [x] `commit` makes submission automatically

## 13. Submission at Google Classroom

- [x] assignment_01.ipynb

## 14. Submission deadline

- 23:30 on the first Sunday after the next class date

## 15. Development Environment 

### Create Python Virtual Environment

```console
$ python -m venv env_XX
```

### Activate a Virtual Environment

```console
$ source env_XX/bin/activate
```

### Install necessary Packages

```console
$ pip install -r requirements.txt
```

### Clone the repository

```console
$ git clone path_to_repository
```

### Edit the codes

- use any development software (e.g. Visual Studio Code)

### Commit the codes

```console
$ git commit -am "commit message"
```

### Push the codes

```console
$ git push origin main
```

### Deactivate a Virtual Environment

```console
$ deactivate
```

### Protected folders
- You should not modify anything under the following protected folders:
- modification will be indicated at GitHub Classroom leading to Fail

```console
.github/workflows
tests
```