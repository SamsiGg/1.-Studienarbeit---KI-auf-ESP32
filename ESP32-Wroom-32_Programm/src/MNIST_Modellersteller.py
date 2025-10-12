import tensorflow as tf
from tensorflow.keras import layers, models

# Lade und prépariere die MNIST-Daten
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Baue ein einfaches CNN-Modell
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(10, activation='softmax') # 10 Outputs für die Ziffern 0-9
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Trainiere das Modell
model.fit(train_images, train_labels, epochs=2)

model.export('mnist_cnn_savedmodel')

converter = tf.lite.TFLiteConverter.from_saved_model("mnist_cnn_savedmodel")

# 3. Führe die Konvertierung durch. Das sollte jetzt funktionieren.
tflite_model = converter.convert()

with open('mnist_cnn.tflite', 'wb') as f:
  f.write(tflite_model)

print("\nModell 'mnist_cnn.tflite' wurde erstellt und ist bereit für den ESP32!")

#xxd -i trained_lstm.tflite > model_data.h