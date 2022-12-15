# 1 import tensorflow as tf
import tensorflow as tf

# 2 save the dataset into a variable
data = tf.keras.datasets.fashion_mnist

# 3 load the dataset into a trainig and validation set
(training_images, training_labels), (validation_images, validation_labels) = data.load_data()

# 4 normalize the data
training_images = training_images / 255.0
validation_images = validation_images / 255.0

# 5 create a model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# 6 compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 7 train the model
model.fit(training_images, training_labels, epochs=5)

# 8 evaluate the model
model.evaluate(validation_images, validation_labels)