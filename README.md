# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks are computational models inspired by the human brain, designed to recognize patterns and relationships in data. In this experiment, we develop a neural network regression model to predict output values based on given inputs. The model consists of one input neuron, two hidden layers with 10 neurons each, and one output neuron. The hidden layers use the ReLU activation function, introducing non-linearity to capture complex patterns. The final layer outputs a continuous value, making the model suitable for regression tasks.

The model is trained using the Mean Squared Error (MSE) loss function, which measures the difference between predicted and actual values. The RMSprop optimizer updates the weights through backpropagation, minimizing the loss over multiple epochs. The dataset is preprocessed using Min-Max Scaling to improve training efficiency.

## Neural Network Model

![image](https://github.com/user-attachments/assets/87dfd82f-d241-4597-a31e-69227b7db7e9)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Arshitha MS
### Register Number: 212223240015
```python
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3=nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.history={'loss':[]}
  def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the Model, Loss Function, and Optimizer
ai_brain=NeuralNet()
criterion= nn.MSELoss()
optimizer=optim.RMSprop(ai_brain.parameters(),lr=0.001)


def train_model(ai_brain,X_train,y_train,criterion,optimizer,epochs=2000):
  for epoch in range(epochs):
    optimizer.zero_grad()
    output=ai_brain(X_train)
    loss=criterion(output,y_train)
    loss.backward()
    optimizer.step()

    ai_brain.history['loss'].append(loss.item())
    if epoch%200==0:
      print(f'epoch {epoch} loss {loss.item():.6f}')
```
## Dataset Information

![image](https://github.com/user-attachments/assets/c118fd4d-ba8c-4284-b10e-c58bd99d9118)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/18b825a7-e572-4fd1-9dd1-b8e346eab266)

### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/43c04e57-2393-467a-8949-03a332a25434)

## RESULT

The trained neural network regression model showed effective learning with a decreasing loss trend and accurate predictions.
