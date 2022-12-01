# importa a biblioteca urllib para buscar um arquivo da web
import urllib.request
import tensorflow.keras as keras
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
