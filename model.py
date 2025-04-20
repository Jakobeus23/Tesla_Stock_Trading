import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras_tuner import Hyperband
from sklearn.metrics import accuracy_score
from tensorflow.keras.regularizers import l2
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input


# Load and preprocess dataset
data = pd.read_csv("tesla_data_other.csv")


# Define the sequence creation function
def create_sequences(data, labels, timesteps):
    Xs, ys = [], []
    for i in range(len(data) - timesteps):
        Xs.append(data[i:(i + timesteps)])
        ys.append(labels[i + timesteps])

    Xs, ys = np.array(Xs), np.array(ys)
    
    # Debugging information
    print("Xs shape:", Xs.shape)  # Expect (num_samples, timesteps, num_features)
    print("ys shape:", ys.shape)  # Expect (num_samples,)
    print("Example sequence (Xs[0]):", Xs[0])
    print("Corresponding label (ys[0]):", ys[0])

    return Xs, ys


# Define the target variable
X = data.drop(columns = ['Price_Increase_7d'])

# response variable
y = data['Price_Increase_7d']


# Define number of timesteps
timesteps = 21

# Initial split into training+validation and test set
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)

# Further split X_temp into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, shuffle=False)  # 15% of the original for validation

# Scale the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Reset indices and create sequences
y_train = y_train.reset_index(drop=True).to_numpy()
y_val = y_val.reset_index(drop=True).to_numpy()
y_test = y_test.reset_index(drop=True).to_numpy()
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, timesteps)
X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, timesteps)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, timesteps)
num_features = X_train_seq.shape[2]


# Define the model for hyperparameter tuning
def build_lstm_model(input_shape, l2_value=0.01):
    model = Sequential()
    
    # Define the input layer
    model.add(Input(shape=input_shape))
    
    # First LSTM layer with L2 regularization
    model.add(LSTM(128, return_sequences=True, kernel_regularizer=l2(l2_value)))
    model.add(Dropout(0.2))  # Dropout for regularization
    
    # Second LSTM layer with L2 regularization
    model.add(LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_value)))
    model.add(Dropout(0.2))
    
    # Third LSTM layer with L2 regularization
    model.add(LSTM(32, kernel_regularizer=l2(l2_value)))
    model.add(Dropout(0.2))
    
    # Output layer with a sigmoid activation for binary classification
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_value)))
    
    # Compile the model with binary cross-entropy loss and Adam optimizer
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Assuming input_shape is (timesteps, num_features)
input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
model = build_lstm_model(input_shape)


history = model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, validation_data=(X_test_seq, y_val_seq))

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test_seq, y_test_seq)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")