
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load model
model = load_model("color_ann_model.keras")

# Load your saved training history
with open("training_history.pkl", "rb") as f:
    history = pickle.load(f)

# Plot accuracy
plt.figure()
plt.plot(history['accuracy'], label='Train Acc')
plt.plot(history['val_accuracy'], label='Val Acc')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accuracy_plot.png")
plt.show()

# Plot loss
plt.figure()
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_plot.png")
plt.show()

# Evaluate on test set
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test, axis=1)

# Identify only the classes present in the test set
import numpy as np

unique_classes = np.unique(y_true)

# Generate classification report 
report = classification_report(
    y_true,
    y_pred,
    labels=unique_classes,
    target_names=label_encoder.inverse_transform(unique_classes)
)
print("Classification Report:\n", report)

# Confusion Matrix 
cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.inverse_transform(unique_classes),
            yticklabels=label_encoder.inverse_transform(unique_classes))
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.show()
