import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback): # Essa classe recebera um parámetro com um método que chama uma função dentro de uma repetição
    def on_epoch_end(self, epoch, logs={}): # a função no caso sera chamada no fim da repetição e verificara se a precisão alcançou 95%
        if(logs.get('accuracy')>0.95):
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback() # a variável callback recebe classe que parara a execução caso o modelo chege a 95% de precisão

data = tf.keras.datasets.fashion_mnist # data fica encarregada dos dados

(training_images, training_labels),(test_images, test_labels) = data.load_data() # usamos o método "load_data" para receber os conjuntos de treino e de teste


# O python permite uma operação feita através do array inteiro de imagens
# no caso a operação divide o valor de cada pixel do array de imagens por 255.0 
# (todas as imagens do conjunto possuem apenas um espectro de cor que vai do branco ao preto "0 a 255")
# isso é feito para gerar um valor "normalizado" entre 0 e 1 que será mais facilmente analizado pela máquina
training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([ # modelo recebe o formato da rede neural utilizado para a aprendizagem
    tf.keras.layers.Flatten(), # Flatten é uma camada de input
    tf.keras.layers.Dense(128, activation=tf.nn.relu), # Dense é uma camada com 128 neuronios que ao serem ativos utilizam uma função que retorna um numero maior que zero
    tf.keras.layers.Dense(10, activation=tf.nn.softmax) # A última camada Dense será a camada de output com uma função que retornara o neuronio com o maior valor
])

model.compile(optimizer='adam', # aqui utilizamos o método "compile" com um parámetro com uma função que "otimiza o aprendizado,
loss='sparse_categorical_crossentropy', # um parametro que calcula a "perda" ou o "erro" de cada repetição
metrics=['accuracy']) # e um parámetro que calcula a precisão de cada repetição do modelo. 

model.fit(training_images, training_labels, # aqui dizemos ao modelo quais os conjuntos utilizado no treinamento
 epochs=50, # e que o modelo deve executar no máximo 50 repetições
 callbacks=[callbacks]) # o parámetro "callbacks" chama a variável "callbacks" que recebeu a classe "myCallback"

# evaluate usa os conjuntos de teste para validar o aprendizado da máquina em um ambiente "desconhecido"
model.evaluate(test_images, test_labels)