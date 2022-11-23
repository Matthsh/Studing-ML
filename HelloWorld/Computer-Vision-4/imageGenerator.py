import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image

training_dir = 'horse-or-human/training/'

# instancia do ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1/255) # All images will be rescaled by 1./255

# setamos alguns hyperparametros para os dados
train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(300, 300),
    class_mode='binary'
)

model = tf.keras.models.Sequential([
    # Recebe o input d uma imagem de 300 x 300 pixels com 3 camadas de cor (rgb), determina 16 filtros com range 3 x 3 e retorna um valor mairo que 0
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    # Corta a imagem em secções de 2 x 2 pixels e filtra o maior valor de cada secção
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Planifica os dados transformando a matriz de pixels em um array
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    # neuronio de retorno com ativação sigmoid, que rotorna 0 ou 1
    tf.keras.layers.Dense(1, activation='sigmoid')  
])

# compila o modelo e seta parámetros e funçoes para o treino
model.compile(
    # função que calcula o erro de cada treino
    loss='binary_crossentropy',
    # função que otimiza o treino (Root Mean Square Propagation = learning rate)
    optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
    # calcula a precisão de cada treino
    metrics=['accuracy']
)

validation_dir = 'horse-or-human/validation/'

validation_datagen = ImageDataGenerator(rescale=1/255)

validation_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size=(300, 300),
    class_mode='binary'
)

# roda o modelo de treino o salva dentro de "history"
history = model.fit_generator(
    train_generator,
    epochs=5,
    validation_data=validation_generator
)

pasta = './content'

for arquivo in os.walk(pasta):
    for fn in arquivo[2]:

        # predictiong images
        path = './content/' + fn
        # carrega a imagem no tamanho correto em que o nodelo foi treinado
        img = tf.keras.utils.load_img(path, target_size=(300, 300))
        # converte a imagem em um array 2D
        x = tf.keras.utils.img_to_array(img)
        # expande as dimençoes do array para 3D como indicado no input_shape
        x = np.expand_dims(x, axis=0)

        # Amontoa o array verticalmente para ficarem no mesmo formato dos dados de treino
        image_tensor = np.vstack([x])
        # salva o array predict dentro de classe
        classes = model.predict(image_tensor)
        # printa o array
        print(classes)
        #printa o primeiro elemento do array
        print(classes[0])
        # se o primeiro elemento do array for maior que 0.5 é humano se não é cavalo
        if classes[0]>0.5:
            print(fn + " is a human")
        else:
            print(fn + " is a horse")
