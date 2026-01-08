import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

model = tf.keras.models.load_model("model/efficientnetb0_mulberry.keras")
datagen = ImageDataGenerator(rescale=1./255)
test_gen = datagen.flow_from_directory("dataset/train", target_size=(224, 224), batch_size=1, shuffle=False)

# Evaluation
print("Evaluating Accuracy...")
results = model.evaluate(test_gen)
print(f"Final Model Accuracy: {results[1]*100:.2f}%")

# Confusion Matrix Logic
Y_pred = model.predict(test_gen)
y_pred = np.argmax(Y_pred, axis=1)
print(classification_report(test_gen.classes, y_pred, target_names=test_gen.class_indices.keys()))