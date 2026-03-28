import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

from data_preprocessing import load_data

# Load data
X_train, y_train, X_test, y_test = load_data()

# Reshape for LSTM (3D input)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Convert labels (1–6 → 0–5)
y_train = y_train - 1
y_test = y_test - 1

# One-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(561, 1)))
model.add(Dense(32, activation='relu'))
model.add(Dense(6, activation='softmax'))

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model (UPDATED EPOCHS)
model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test)
)

# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)

# Save model (UPDATED FORMAT)
model.save("activity_model.keras")