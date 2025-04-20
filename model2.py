from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.metrics import AUC
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("Tesla_data_original.csv")
df.drop(columns=['Volume', 'Target_Close_7d'], inplace=True, errors='ignore')

X = df.drop(columns=['Price_Increase_7d'])
y = df['Price_Increase_7d']

# Min-max scale the full feature set
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Define sequence creation function
def create_lstm_sequences(X, y, timesteps=21):
    X_seq, y_seq = [], []
    for i in range(len(X) - timesteps):
        X_seq.append(X[i:i + timesteps])
        y_seq.append(y[i + timesteps])
    return np.array(X_seq), np.array(y_seq)

# Apply sequence creation
timesteps = 3
X_seq, y_seq = create_lstm_sequences(X_scaled, y.values, timesteps=timesteps)

# Now split into train and test

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=69)
for train_idx, test_idx in splitter.split(X_seq, y_seq):
    X_train, X_test = X_seq[train_idx], X_seq[test_idx]
    y_train, y_test = y_seq[train_idx], y_seq[test_idx]

# Define extreme class weight to force focus on class 1
class_weight = {0: 1.0, 1: 1.0}

# Build the LSTM model
input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)

LSTM_model = Sequential()
LSTM_model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
LSTM_model.add(Dropout(0.1))
LSTM_model.add(LSTM(64, return_sequences=True))
LSTM_model.add(Dropout(0.2))
LSTM_model.add(LSTM(32, return_sequences=False))
LSTM_model.add(Dropout(0.3))
LSTM_model.add(Dense(1, activation='sigmoid'))
LSTM_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[AUC(name='auc')])



# Train the model
history = LSTM_model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,  # Good match for your timestep length
    validation_data=(X_test, y_test),
    class_weight=class_weight
)

from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Evaluate the model
test_loss, test_auc = LSTM_model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test AUC: {test_auc}")

# Predict on test data
y_pred = LSTM_model.predict(X_test).flatten()
y_pred_binary = (y_pred > 0.5).astype(int)

# Accuracy & F1
acc = accuracy_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)
print(f"\nAccuracy Score: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_binary))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_binary)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Wrap y_test into a Series
y_test_indexed = pd.Series(y_test)

# Identify incorrect predictions
incorrect_mask = y_test_indexed != y_pred_binary
incorrect_df = pd.DataFrame({
    'Actual': y_test_indexed[incorrect_mask].values,
    'Predicted': y_pred_binary[incorrect_mask],
    'Probability': y_pred[incorrect_mask]
})

print(f"\nNumber of incorrect predictions: {len(incorrect_df)}")
print(incorrect_df)

incorrect_df.to_csv("incorrect_predictions.csv", index=False)


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred_binary)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Calculate false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot the AUC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

