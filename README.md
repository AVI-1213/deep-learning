# Deep-learning Report : Predicting Funding Success

## Overview

This project aims to develop a deep learning model to predict the success of funding applicants for the nonprofit organization Alphabet Soup. The model uses historical data of funded organizations to create a binary classifier that identifies likely successful applicants.

## Dataset

The dataset contains metadata about over 34,000 organizations that have received funding, including the following columns:

- `EIN`: Identification number
- `NAME`: Organization name
- `APPLICATION_TYPE`: Type of application submitted
- `AFFILIATION`: Affiliated sector of industry
- `CLASSIFICATION`: Government organization classification
- `USE_CASE`: Purpose for funding
- `ORGANIZATION`: Type of organization
- `STATUS`: Active status of the organization
- `INCOME_AMT`: Income classification
- `SPECIAL_CONSIDERATIONS`: Any special considerations
- `ASK_AMT`: Amount of funding requested
- `IS_SUCCESSFUL`: Indicator of whether the funding was used successfully

## Data Preprocessing

- **Target Variable:**
  - `IS_SUCCESSFUL`: Indicates if the funding was used effectively.

- **Feature Variables:**
  - `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, `ASK_AMT`, `NAME` (Optimized Model)

- **Removed Variables:**
  - `EIN` (Both models), `NAME` (Initial Model)

## Neural Network Model

### Initial Model

- **Architecture:**
  - Input Layer: Number of neurons equal to the number of input features
  - First Hidden Layer: 8 neurons with 'relu' activation
  - Second Hidden Layer: 4 neurons with 'relu' activation
  - Output Layer: 1 neuron with 'sigmoid' activation
```Python
number_input_features = len(X_train[0])
hidden_nodes_layer1 =  8
hidden_nodes_layer2 = 4
nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(
    tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu"))

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
```
- **Performance:**
  - Achieved an accuracy of approximately 73%, with loss of 0.555.
```Python
model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

268/268 - 0s - 696us/step - accuracy: 0.7300 - loss: 0.5553
Loss: 0.5553125143051147, Accuracy: 0.7300291657447815
```

### Optimized Model

- **Architecture:**
  - Input Layer: Number of neurons equal to the number of input features
  - First Hidden Layer: 8 neurons with 'relu' activation
  - Second Hidden Layer: 16 neurons with 'relu' activation
  - Third Hidden Layer: 21 neurons with 'relu' activation
  - Output Layer: 1 neuron with 'sigmoid' activation
```Python
number_input_features = len(X_train[0])
hidden_nodes_layer1 =  8
hidden_nodes_layer2 = 16
hidden_nodes_layer3 = 21

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(
    tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu")
)
# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))
# Third hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer3, activation="relu"))
# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
```

- **Performance:**
  - Achieved an accuracy of 78.33%, with a loss of 0.4684.
```Python
268/268 - 0s - 433us/step - accuracy: 0.7833 - loss: 0.4684
Loss: 0.46843841671943665, Accuracy: 0.7833235859870911
```
## Summary

The deep learning model achieved a significant improvement in predicting the success of funding applicants. By including additional features variables 'NAME' and refining the network architecture and hidden layers, the optimized model reached an accuracy of 78.33%. Future work could explore alternative models like using 'Hyperparameter -Hyperband / Gridsearch tuner'and further feature engineering to enhance prediction accuracy.

## Files

- `AlphabetSoupCharity.h5`: Model weights from the initial neural network.
- `AlphabetSoupCharity_Optimized.h5`: Model weights from the optimized neural network.

## Dependencies
```
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import tensorflow as tf

import pandas as pd

```
## Author

- [Avinash] - [[GitHub Profile](https://github.com/AVI-1213)]
