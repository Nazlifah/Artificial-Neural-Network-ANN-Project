import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical
import pickle

# Load and clean dataset
df = pd.read_csv("dataset.csv", encoding='ISO-8859-1')
df = df.drop(df.columns[0], axis=1)
df.columns = ['color_name', 'hex', 'R', 'G', 'B']
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df[['R', 'G', 'B']] = df[['R', 'G', 'B']].astype(int)

# Prepare data
X = df[['R', 'G', 'B']].values
y = df['color_name'].values
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Build ANN model with Input layer
model = Sequential()
model.add(Input(shape=(3,)))                  
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(y_categorical.shape[1], activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train and capture history
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.1,
    verbose=1
)

# Save test sets for later evaluation
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

# Save the training history
with open("training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Save model and encoder
model.save("color_ann_model.keras")  
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)


