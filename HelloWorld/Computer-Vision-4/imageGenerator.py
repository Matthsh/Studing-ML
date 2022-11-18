import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

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
    # Recebe o input d uma imagem de 300 x 300 pixels com 3 camadas de cor, determina 16 filtros com range 3 x 3 e retorna um valor mairo que 0
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


model.compile(
    # função que calcula o erro de cada treino
    loss='binary_crossentropy',
    # função que otimiza o treino (Root Mean Square Propagation = learning rate)
    optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
    # calcula a precisão de cada treino
    metrics=['accuracy']
)


history = model.fit(
    train_generator,
    epochs=15
)

