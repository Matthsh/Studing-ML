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

# atribuimos hyperparametros a um modelo pré-treinado do google "InceptionV3"
pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
 include_top=False,
 weights=None)
pre_trained_model.load_weights(weights_file)

for layer in pre_trained_model.layers:
 layer.trainable = False
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

for layer in pre_trained_model.layers:
 layer.trainable = False
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

for layer in pre_trained_model.layers:
 layer.trainable = False
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output
