# importa a biblioteca urllib para buscar um arquivo da web
import urllib.request
import keras
from keras import layers, optimizers
from keras.applications.inception_v3 import InceptionV3

# atribui o url dos pessos a "weights_url"
weights_url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
# atribui o nome do arquivo a ser buscado a "weights_file"
weights_file = "inception_v3.h5"

# usa a bilbioteca urllib para buscar(recuperar) no url de "weights_url" o arquivo de "weights_file"
urllib.request.urlretrieve(weights_url, weights_file)

# salva um modelo pré-treinado do google chamado "InceptionV3" especificando um formato para ele
pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
 include_top=False,
 weights=None)
# carrega o arquivo de pesos no modelo pré-treinado
pre_trained_model.load_weights(weights_file)

# para cada camada no modelo pré-treinado
for layer in pre_trained_model.layers:
 # o parametro "trainable" recebe falso, significando que a camada não será mais treinada (eu acho)
 layer.trainable = False
# salva a camada 'mixed7' do modelo pré-treinado na variável "last_layer"
last_layer = pre_trained_model.get_layer('mixed7')
# printa o formato da ultima camada(last_layer) do modelo
print('last layer output shape: ', last_layer.output_shape)
# salva o formato de saida da ultima camada na variável "last_output"
last_output = last_layer.output

# Salva em x uma camada planificada(de 1 dimensão) originada de "last_output"
x = layers.Flatten()(last_output)
# Salva em x outra camada, com 1024 neurônios e ativação relu que se soma a camada x anterior
x = layers.Dense(1024, activation='relu')(x)
# Salva em x uma última camada, sigmoid, que gera o output.
x = layers.Dense(1, activation='sigmoid')(x)

# define o modelo de treino com a camada de input do modelo pré-treinado do google e as camadas salvas em x.
model = keras.Model(pre_trained_model.input, x)
# compila o modelo com o otimizador RMSprop(Root Mean Square Propagation), algoritimo de perda de entropia cruzada binária e métrica acc
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0001),
 loss='binary_crossentropy',
 metrics=['acc'])

# importa a biblioteca os para manipular arquivos
import os
# importa a biblioteca zipfile para manipular arquivos zip
import zipfile

# atribui o url dos pessos a "local_zip"
local_zip = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
# atribui o nome do arquivo a ser buscado a "zip_file"
zip_file = 'cats_and_dogs_filtered.zip'
# usa a bilbioteca urllib para buscar(recuperar) no url de "local_zip" o arquivo de "zip_file"
urllib.request.urlretrieve(local_zip, zip_file)

# extrai o arquivo zipado
zip_ref = zipfile.ZipFile(zip_file, 'r')
zip_ref.extractall()
zip_ref.close()

# define o nome do diretório base
base_dir = 'cats_and_dogs_filtered'

# define o nome do diretório de treino
train_dir = os.path.join(base_dir, 'train')
# define o nome do diretório de validação
validation_dir = os.path.join(base_dir, 'validation')

# define o nome do diretório de treino de gatos
train_cats_dir = os.path.join(train_dir, 'cats')
# define o nome do diretório de treino de cachorros
train_dogs_dir = os.path.join(train_dir, 'dogs')
# define o nome do diretório de validação de gatos
validation_cats_dir = os.path.join(validation_dir, 'cats')
# define o nome do diretório de validação de cachorros
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# importa a biblioteca ImageDataGenerator para gerar dados de imagens
from keras.preprocessing.image import ImageDataGenerator

# define o tamanho das imagens de treino e validação
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# define o tamanho das imagens de treino e validação
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=20, class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=20, class_mode='binary')

# treina o modelo com 100 épocas, 100 passos por época e 50 passos de validação por época
history = model.fit(train_generator, epochs=3, validation_data=validation_generator)

# importa a biblioteca matplotlib para plotar gráficos
import matplotlib.pyplot as plt

# define o nome dos eixos x e y
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# define o tamanho do gráfico
plt.figure(figsize=(8, 8))
# define o número de linhas e colunas do gráfico
plt.subplot(2, 1, 1)
# plota o gráfico de acurácia
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

# plota o gráfico de perda
plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()