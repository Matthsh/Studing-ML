# importa a biblioteca ImageDataGenerator para gerar dados de imagens
from keras.preprocessing.image import ImageDataGenerator
# importa a biblioteca tensorflow
import tensorflow as tf


# define o nome do diretório base
TRAINING_DIR = "/home/matthsh/Documentos/Studing-ML/HelloWorld/Computer-Vision-6/training-dataset/rps"
# define o nome do diretório de treino e alguns parametros de formatação para as imagens
training_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest'
)


# gera os dados de treino
train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150,150),
    class_mode='categorical'
)


# define o modelo de rede neural
model = tf.keras.models.Sequential([
    
    # primeira camada de convolução
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    # segunda camada de convolução
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # terceira camada de convolução
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),

    # quarta camada de convolução
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # camada de achatamento dos resultados para alimentar o DNN (Deep Neural Network)
    tf.keras.layers.Flatten(),

    # camada escondida de 512 neurônios
    tf.keras.layers.Dense(512, activation='relu'),

    # camada de saída com 3 neurônios
    tf.keras.layers.Dense(3, activation='softmax')
])


# compila o modelo
model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# treina o modelo
history = model.fit(train_generator, epochs=25, steps_per_epoch=20, verbose=1)


# salva o modelo dentro da pasta /modelo-history
model.save("/home/matthsh/Documentos/Studing-ML/HelloWorld/Computer-Vision-6/modelo-history/modelo.h5")

# salva o histórico de treinamento dentro da pasta /modelo-history
import pickle
with open('/home/matthsh/Documentos/Studing-ML/HelloWorld/Computer-Vision-6/modelo-history/history.pickle', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
