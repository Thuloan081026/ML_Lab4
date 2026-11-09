# Training Deep Neural Networks with PyTorch

> Study notes based on **Chapter 3 – Training Deep Neural Networks**  
> Lecturer: **PhD. Nguyễn Thị Khánh Tiên**  
> Trường Đại học Giao thông Vận tải TP.HCM – Viện Công nghệ Thông tin, Điện, Điện tử

---

## Table of Contents

1. [Introduction to Deep Neural Networks](#1-introduction-to-deep-neural-networks)
2. [DNN Training Process](#2-dnn-training-process)
3. [Challenges in Training DNNs](#3-challenges-in-training-dnns)
4. [Training a DNN with PyTorch](#4-training-a-dnn-with-pytorch)
5. [Prepare Data: Dataset and DataLoader](#5-prepare-data-dataset-and-dataloader)
6. [Define the Model](#6-define-the-model)
7. [Loss Function and Optimizer](#7-loss-function-and-optimizer)
8. [Training Loop](#8-training-loop)
9. [Validation](#9-validation)
10. [Save and Load the Model](#10-save-and-load-the-model)
11. [TensorBoard](#11-tensorboard)
12. [Vanishing and Exploding Gradients](#12-vanishing-and-exploding-gradients)
13. [Weight Initialization](#13-weight-initialization)
14. [Activation Functions](#14-activation-functions)
15. [Batch Normalization](#15-batch-normalization)
16. [Important Considerations](#16-important-considerations)
17. [Transfer Learning](#17-transfer-learning)
18. [Fine-tuning](#18-fine-tuning)
19. [Regularization](#19-regularization)
20. [Practice: Pre-trained Neural Networks](#20-practice-pre-trained-neural-networks)
21. [Quick Summary](#21-quick-summary)

---

# 1. Introduction to Deep Neural Networks

## What is a DNN?

A **Deep Neural Network (DNN)** is a computer model constructed from multiple layers of artificial neurons, designed to mimic how the human brain processes information.

The term **"Deep"** refers to the number of hidden layers in the network.

- More hidden layers → the network can learn more complex patterns.
- DNNs are trained using data to perform a specific task.

## What is DNN Training?

Training a DNN is the process of teaching a computer model to perform a specific task using data.

The model learns by adjusting its parameters, especially its weights, so that its predictions become closer to the expected outputs.

---

# 2. DNN Training Process

The main components of DNN training are:

### 2.1 Data

A large amount of **labeled data** is needed to train the model.

The data provides examples from which the neural network learns patterns.

### 2.2 Algorithms

The most common algorithm for training DNNs is **backpropagation**.

Backpropagation adjusts the network's weights to minimize the error between:

- predicted output
- actual output

### 2.3 Optimization

Optimization algorithms are used to find appropriate values for the network's weights.

Common optimizers include:

- **SGD** – Stochastic Gradient Descent
- **Adam**
- **RMSprop**

### 2.4 Iteration

Training is repeated many times using **epochs**.

Each epoch allows the model to learn patterns more accurately from the training data.

---

# 3. Challenges in Training DNNs

## 3.1 Overfitting

**Overfitting** occurs when the model learns the training data too well but performs poorly on new, unseen data.

In other words:

```text
Training performance  → good
New/unseen data       → poor
```

Validation data can be used to monitor this problem during training.

---

## 3.2 Vanishing Gradients

Vanishing gradients occur when gradients become too small during backpropagation.

As a result:

- weights in early layers update very slowly;
- learning becomes ineffective;
- training deep networks becomes difficult.

This problem is common with activation functions such as:

- Sigmoid
- Tanh

---

## 3.3 Exploding Gradients

Exploding gradients occur when gradients become too large.

This can lead to:

- unstable weight updates;
- unstable training;
- potentially breaking the training process.

It is common in:

- deep networks;
- recurrent neural networks (RNNs).

---

## 3.4 High Computational Requirements

Training DNNs can require significant computing power, especially when:

- the network is large;
- the dataset is large.

A GPU can significantly speed up DNN training.

---

# 4. Training a DNN with PyTorch

PyTorch provides tools for constructing and training neural networks.

The fundamental training steps are:

```text
1. Prepare Your Data
       ↓
2. Define Your Model
       ↓
3. Choose Loss Function & Optimizer
       ↓
4. Write Training Loop
       ↓
5. Validation
       ↓
6. Save / Load Model
```

---

# 5. Prepare Data: Dataset and DataLoader

PyTorch provides:

- `Dataset`
- `DataLoader`

to efficiently manage data.

## 5.1 Dataset

A `Dataset` is responsible for:

- storing samples;
- storing corresponding labels;
- implementing `__len__()`;
- implementing `__getitem__()`.

### `__len__()`

Returns the size of the dataset.

### `__getitem__()`

Returns a sample and its label at a specified index.

---

## 5.2 Custom Dataset

Usually, a custom dataset class inherits from:

```python
torch.utils.data.Dataset
```

Example structure:

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
```

The exact implementation depends on the format of the dataset.

---

## 5.3 DataLoader

`DataLoader` is responsible for:

- iterating through the dataset in batches;
- shuffling data;
- loading data in parallel using multiple workers.

Example:

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)
```

### Why use batches?

Instead of sending the entire dataset to the model at once, the data is divided into smaller batches.

```text
Dataset
   ↓
Batch 1
Batch 2
Batch 3
...
Batch N
```

This makes training more manageable.

---

## 5.4 Common Datasets

For common datasets such as:

- MNIST
- CIFAR-10

PyTorch's `torchvision.datasets` provides pre-built dataset classes.

---

# 6. Define the Model

A neural network architecture in PyTorch is normally defined by inheriting from:

```python
torch.nn.Module
```

A model generally defines two important methods:

- `__init__(self)`
- `forward(self, x)`

---

## 6.1 `__init__()`

The `__init__()` method defines the layers of the network.

Common layers include:

```python
nn.Linear
nn.Conv2d
nn.ReLU
nn.MaxPool2d
```

---

## 6.2 `forward()`

The `forward()` method defines how data flows through the network.

Example:

```python
import torch.nn as nn

class SimpleDNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
```

The basic flow is:

```text
Input
  ↓
Linear Layer
  ↓
ReLU
  ↓
Linear Layer
  ↓
Output
```

---

# 7. Loss Function and Optimizer

Training requires two important components:

1. **Loss Function**
2. **Optimizer**

---

## 7.1 Loss Function

The loss function measures how well the model is performing.

The appropriate loss function depends on the task.

### Multi-class Classification

Use:

```python
nn.CrossEntropyLoss()
```

### Binary Classification

Common choices:

```python
nn.BCELoss()
```

or:

```python
nn.BCEWithLogitsLoss()
```

### Regression

Use:

```python
nn.MSELoss()
```

---

## 7.2 Optimizer

The optimizer updates the model's parameters:

- weights;
- biases.

Its objective is to minimize the loss.

Common optimizers:

### SGD

```python
torch.optim.SGD
```

### Adam

```python
torch.optim.Adam
```

Adam is often a good default choice.

### RMSprop

```python
torch.optim.RMSprop
```

Example:

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

---

# 8. Training Loop

The training loop typically iterates over:

- epochs;
- batches.

General structure:

```text
For each epoch:
    For each batch:
        Move data to device
        Clear old gradients
        Forward pass
        Calculate loss
        Backward pass
        Update parameters
```

---

## 8.1 Basic PyTorch Training Loop

```python
for epoch in range(num_epochs):

    model.train()

    for data, labels in train_loader:

        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(data)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()
```

---

## 8.2 Step 1: Move Data to Device

If GPU is available, data and model can be moved to the GPU.

```python
data = data.to(device)
labels = labels.to(device)
```

The device can be:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

---

## 8.3 Step 2: Zero Gradients

PyTorch accumulates gradients by default.

Therefore, gradients from the previous batch must be cleared:

```python
optimizer.zero_grad()
```

---

## 8.4 Step 3: Forward Pass

The input data is passed through the model:

```python
outputs = model(data)
```

The result is the model's prediction.

---

## 8.5 Step 4: Calculate Loss

Compare predictions with the true labels:

```python
loss = criterion(outputs, labels)
```

The loss represents the difference between the prediction and the expected output.

---

## 8.6 Step 5: Backward Pass

Calculate gradients:

```python
loss.backward()
```

This is the **backpropagation** step.

It computes the gradients of the loss with respect to the model's parameters:

```text
Loss
 ↓
Backpropagation
 ↓
Gradients
 ↓
Weights / Biases
```

---

## 8.7 Step 6: Optimization Step

Update the parameters:

```python
optimizer.step()
```

The optimizer uses the calculated gradients to update the model parameters.

---

# 9. Validation

Validation is important for evaluating the model on a separate dataset during training.

It helps:

- monitor overfitting;
- choose the best model;
- tune hyperparameters.

Examples of hyperparameters include:

- learning rate;
- regularization;
- batch size;
- number of epochs.

---

## 9.1 `model.eval()`

During validation:

```python
model.eval()
```

This puts the model into evaluation mode.

This is especially important for layers such as:

- Batch Normalization
- Dropout

because they behave differently during training and evaluation.

---

## 9.2 `torch.no_grad()`

During validation:

```python
with torch.no_grad():
    ...
```

This disables gradient calculations.

Benefits:

- faster validation;
- lower memory usage.

---

## 9.3 Validation Loop

Example:

```python
model.eval()

with torch.no_grad():

    for data, labels in val_loader:

        data = data.to(device)
        labels = labels.to(device)

        outputs = model(data)

        loss = criterion(outputs, labels)
```

After validation, switch back to training mode:

```python
model.train()
```

---

# 10. Save and Load the Model

After training, the learned parameters can be saved so that the model can be reused without retraining.

Typical PyTorch workflow:

```text
Training
   ↓
Learned Parameters
   ↓
Save
   ↓
Model File
   ↓
Load Later
   ↓
Reuse Model
```

Example:

```python
torch.save(model.state_dict(), "model.pth")
```

Loading:

```python
model.load_state_dict(
    torch.load("model.pth")
)
```

The model architecture must be defined before loading its parameters.

---

# 11. TensorBoard

**TensorBoard** is a tool for visualizing:

- model training;
- loss;
- accuracy;
- learning rate;
- model architecture;
- images;
- weights;
- activations;
- embeddings.

## Advantages

TensorBoard provides:

- real-time visualization;
- easier debugging;
- model comparison;
- data exploration;
- result sharing.

---

## 11.1 Installation

Install the required packages:

```bash
pip install tensorboard torch torchvision
```

---

## 11.2 SummaryWriter

Import:

```python
from torch.utils.tensorboard import SummaryWriter
```

Create a writer:

```python
writer = SummaryWriter("runs/experiment_1")
```

---

## 11.3 Visualizing Scalars

Scalar values such as:

- loss;
- accuracy;
- learning rate

can be logged with:

```python
writer.add_scalar()
```

Example:

```python
writer.add_scalar(
    "Loss/train",
    loss,
    epoch
)
```

---

## 11.4 Visualizing Model Architecture

Use:

```python
writer.add_graph()
```

A dummy input is needed to trace the model.

---

## 11.5 Visualizing Images

Images from:

- datasets;
- predictions

can be added using:

```python
writer.add_image()
```

---

## 11.6 Visualizing Histograms

Weights or activations can be visualized with:

```python
writer.add_histogram()
```

This can help inspect their distributions.

---

## 11.7 Visualizing Embeddings

High-dimensional data such as word embeddings can be visualized in a lower-dimensional space using:

```python
writer.add_embedding()
```

---

## 11.8 Running TensorBoard

Run:

```bash
tensorboard --logdir=runs
```

Then open:

```text
http://localhost:6006/
```

---

# 12. Vanishing and Exploding Gradients

## 12.1 Vanishing Gradients

Vanishing gradients happen when gradients become too small during backpropagation.

Consequences:

```text
Gradient becomes very small
        ↓
Early-layer weights update very little
        ↓
Learning becomes ineffective
```

Common activation functions associated with this problem:

- Sigmoid
- Tanh

---

## 12.2 Exploding Gradients

Exploding gradients happen when gradients become excessively large.

Consequences:

```text
Gradient becomes very large
        ↓
Large weight updates
        ↓
Unstable training
```

---

## 12.3 Solutions

Possible solutions mentioned in the material include:

### Use ReLU or its variants

```python
nn.ReLU()
```

or:

```python
nn.LeakyReLU()
```

### Gradient Clipping

Limit excessively large gradients.

### Careful Weight Initialization

Initialize weights appropriately.

### Batch Normalization

Normalize layer outputs to help stabilize training.

---

# 13. Weight Initialization

## Purpose

Weights should be initialized appropriately to promote:

- stable training;
- efficient training.

---

## 13.1 Random Initialization

Weights can be initialized using a random distribution such as:

- normal distribution;
- uniform distribution.

---

## 13.2 Xavier / Glorot Initialization

Xavier initialization considers the number of:

- inputs;
- outputs

of a layer.

Its purpose is to balance gradients during backpropagation.

---

## 13.3 He Initialization

He initialization is similar to Xavier initialization but is optimized for networks using **ReLU** activation functions.

---

## 13.4 PyTorch Implementation

PyTorch provides:

```python
torch.nn.init
```

for weight initialization.

---

# 14. Activation Functions

## Purpose

Activation functions introduce **non-linearity** into neural networks.

Without non-linearity, the network would have limited ability to learn complex relationships.

---

## 14.1 ReLU

Formula:

```text
ReLU(x) = max(0, x)
```

Characteristics:

- simple;
- efficient;
- helps mitigate vanishing gradients.

PyTorch:

```python
nn.ReLU()
```

---

## 14.2 Sigmoid

The output is between:

```text
0 and 1
```

It is often used for:

- binary classification.

PyTorch:

```python
nn.Sigmoid()
```

---

## 14.3 Tanh

Output range:

```text
-1 to 1
```

PyTorch:

```python
nn.Tanh()
```

---

## 14.4 LeakyReLU

LeakyReLU improves ReLU by allowing a small gradient when the input is negative.

PyTorch:

```python
nn.LeakyReLU()
```

---

# 15. Batch Normalization

## Purpose

Batch Normalization is used to:

- normalize layer outputs;
- mitigate vanishing gradients;
- mitigate exploding gradients;
- accelerate training.

---

## How It Works

The layer outputs are normalized toward a standard distribution:

```text
Mean = 0
Variance = 1
```

It then applies:

- scale;
- shift

parameters so that the network can learn an appropriate distribution.

---

## PyTorch

For different types of data:

```python
nn.BatchNorm1d
nn.BatchNorm2d
nn.BatchNorm3d
```

---

# 16. Important Considerations

## 16.1 CPU vs GPU

For real-world DNN training, a GPU can be important for training speed.

Typical device configuration:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

Then move the model to the selected device:

```python
model = model.to(device)
```

and data:

```python
data = data.to(device)
```

---

## 16.2 Hyperparameter Tuning

Important hyperparameters include:

- learning rate;
- batch size;
- number of epochs;
- network architecture;
- number of layers;
- number of hidden units;
- regularization parameters.

These values generally need to be experimented with and tuned for the specific dataset and problem.

---

## 16.3 More Complex Models

Different types of data may require different neural network architectures.

### Image Data

Use **Convolutional Neural Networks (CNNs)**.

Common layers:

```python
nn.Conv2d
nn.MaxPool2d
```

### Sequence Data

Examples include:

- text;
- time series.

Possible architectures:

- RNNs;
- Transformers.

---

# 17. Transfer Learning

## Purpose

Transfer learning leverages knowledge from a model pre-trained on a large dataset.

An example is a model pre-trained on:

```text
ImageNet
```

This can be useful when the target dataset is relatively small.

---

## Steps

### Step 1: Load a Pre-trained Model

PyTorch provides pre-trained models through:

```python
torchvision.models
```

---

### Step 2: Freeze Layers

Freeze the layers of the pre-trained model so their weights are not updated during retraining.

Conceptually:

```text
Pre-trained model
       ↓
Freeze existing layers
       ↓
Keep learned features
```

---

### Step 3: Replace or Add Layers

The final classification layer can be replaced so that it matches the number of classes in the target dataset.

Additional layers can also be added for customization.

---

### Step 4: Retrain

Retrain the model on the new dataset.

Only the weights of the unfrozen layers are updated.

---

# 18. Fine-tuning

## Purpose

Fine-tuning refines the weights of a pre-trained model using the target dataset.

---

## Steps

### Step 1: Load the Pre-trained Model

Similar to transfer learning.

### Step 2: Unfreeze Layers

Some final layers are unfrozen so their weights can be updated.

### Step 3: Retrain

Retrain the model using the target dataset.

---

## Transfer Learning vs Fine-tuning

The material distinguishes them mainly by which layers are allowed to update:

```text
Transfer Learning
    ↓
Freeze layers
    ↓
Train selected new/unfrozen layers


Fine-tuning
    ↓
Unfreeze some final layers
    ↓
Update those layers during retraining
```

Both approaches are particularly useful when the available dataset is small.

---

# 19. Regularization

## Purpose

Regularization is used to help prevent **overfitting**.

The material describes regularization as adding a penalty to the loss function.

Hyperparameters such as:

- learning rate;
- dropout rate;
- regularization strength

can be experimented with for a specific dataset.

---

# 19.1 Dropout

Dropout randomly deactivates neurons during training.

Purpose:

- prevent co-adaptation of neurons;
- reduce overfitting.

PyTorch:

```python
nn.Dropout()
```

Important:

```text
Training → Dropout is applied
Evaluation → Dropout is not applied
```

---

# 19.2 L1 Regularization

L1 regularization adds the sum of the absolute values of the weights to the loss function.

Conceptually:

```text
L1 penalty = sum(|weight|)
```

Effects:

- encourages sparse weights;
- many weights can become zero;
- useful for feature selection.

---

# 19.3 L2 Regularization

L2 regularization adds the sum of squared weights to the loss function.

Conceptually:

```text
L2 penalty = sum(weight²)
```

Effects:

- encourages small weights;
- reduces the impact of outliers.

---

## 19.4 L1 vs L2

| Technique | Penalty | Main effect |
|---|---|---|
| L1 | `sum(abs(weights))` | Encourages sparse weights |
| L2 | `sum(weights²)` | Encourages small weights |

### PyTorch

L2 regularization can be applied using:

```python
weight_decay
```

in the optimizer.

Example:

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)
```

For L1 regularization, the L1 penalty needs to be manually added to the loss.

---

# 20. Practice: Pre-trained Neural Networks

The practice section focuses on hands-on work with pre-trained neural network architectures.

## Exercises

1. Experiment with different pre-trained models.
2. Adjust hyperparameters such as:
   - learning rate;
   - batch size.
3. Use TensorBoard to monitor training.

Examples of models:

- VGG
- DenseNet
- ResNet

---

## Objective

The objectives are to:

- become familiar with loading and using pre-trained models from `torchvision.models`;
- understand how to adapt pre-trained models;
- understand how to fine-tune models for specific tasks.

---

## Practice Workflow

```text
Environment Setup
       ↓
Load Pre-trained Model
       ↓
Explore Model Architecture
       ↓
Adapt Model
       ↓
Prepare Data
       ↓
Train Model
       ↓
Evaluate Model
```

---

## Step 1: Environment Setup

Make sure PyTorch and torchvision are installed:

```bash
pip install torch torchvision
```

---

## Step 2: Load a Pre-trained Model

`torchvision.models` provides models such as:

- ResNet;
- VGG;
- DenseNet.

The material describes loading models with weights pre-trained on the ImageNet dataset.

---

## Step 3: Explore Model Architecture

Print the model to examine:

- structure;
- layers;
- parameters.

Example:

```python
print(model)
```

---

## Step 4: Adapt the Model

The final classification layer normally needs to be replaced to match the number of classes in the target dataset.

Two approaches can be used:

### Transfer Learning

Freeze earlier layers:

```text
Earlier layers → Frozen
Final layers   → Train
```

### Fine-tuning

Unfreeze some final layers:

```text
Earlier layers → Frozen
Final layers   → Trainable
```

---

## Step 5: Prepare the Data

Prepare the dataset and preprocess the data.

For image data, `torchvision.transforms` can be used for preprocessing.

Then create batches using:

```python
torch.utils.data.DataLoader
```

---

## Step 6: Train the Model

Define:

- loss function;
- optimizer;
- training loop.

The training loop updates the trainable model parameters.

---

## Step 7: Evaluate the Model

Evaluate the model's performance on the test dataset.

---

# 21. Quick Summary

## Complete DNN Training Pipeline

```text
                  ┌──────────────────┐
                  │      DATA        │
                  └────────┬─────────┘
                           ↓
                  ┌──────────────────┐
                  │ Dataset /        │
                  │ DataLoader       │
                  └────────┬─────────┘
                           ↓
                  ┌──────────────────┐
                  │ Define Model     │
                  │ nn.Module        │
                  └────────┬─────────┘
                           ↓
                  ┌──────────────────┐
                  │ Loss Function    │
                  │ + Optimizer      │
                  └────────┬─────────┘
                           ↓
                  ┌──────────────────┐
                  │ Training Loop    │
                  │ Forward          │
                  │ Loss             │
                  │ Backpropagation  │
                  │ Optimizer Step   │
                  └────────┬─────────┘
                           ↓
                  ┌──────────────────┐
                  │   Validation     │
                  └────────┬─────────┘
                           ↓
                  ┌──────────────────┐
                  │ Save Best Model  │
                  └────────┬─────────┘
                           ↓
                  ┌──────────────────┐
                  │ Test / Reuse     │
                  └──────────────────┘
```

---

## Core PyTorch Concepts

| Concept | Purpose |
|---|---|
| `Dataset` | Stores samples and labels |
| `DataLoader` | Creates batches and loads data |
| `nn.Module` | Defines the neural network |
| `forward()` | Defines data flow through the model |
| Loss Function | Measures prediction error |
| Optimizer | Updates model parameters |
| `loss.backward()` | Computes gradients |
| `optimizer.step()` | Updates weights |
| `model.train()` | Training mode |
| `model.eval()` | Evaluation mode |
| `torch.no_grad()` | Disables gradient computation |
| `torch.save()` | Saves model parameters |
| `torch.load()` | Loads saved parameters |
| TensorBoard | Visualizes training |
| Dropout | Regularization |
| Batch Normalization | Normalizes layer outputs |
| Transfer Learning | Reuses a pre-trained model |
| Fine-tuning | Updates selected pre-trained layers |

---

## Core Training Logic

The most important training sequence to remember is:

```python
optimizer.zero_grad()

outputs = model(data)

loss = criterion(outputs, labels)

loss.backward()

optimizer.step()
```

The logic is:

```text
Clear old gradients
        ↓
Forward Pass
        ↓
Calculate Loss
        ↓
Backpropagation
        ↓
Update Weights
```

---

## Key Problems and Solutions

| Problem | Description | Solutions mentioned |
|---|---|---|
| Overfitting | Good training performance but poor unseen-data performance | Validation, regularization |
| Vanishing Gradient | Gradients become too small | ReLU/variants, careful initialization, BatchNorm |
| Exploding Gradient | Gradients become too large | Gradient clipping, careful initialization, BatchNorm |
| High computation | Large models/datasets require substantial computing power | GPU |

---

## References

- Course material: **Chapter 3 – Training Deep Neural Networks**, PhD. Nguyễn Thị Khánh Tiên.
- PyTorch TensorBoard tutorial: https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
- NVIDIA PyTorch glossary: https://www.nvidia.com/en-au/glossary/pytorch/

---

## Notes

This README is organized from the provided course material and keeps its main terminology, structure, and concepts. It is intended as a study/reference document for learning how to train Deep Neural Networks with PyTorch.
