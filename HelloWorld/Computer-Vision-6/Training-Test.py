# Modelo de predição de imagens

import os
import tensorflow as tf
import numpy as np

class_names = ['papel', 'pedra', 'tesoura']

# atribui o diretório de amostras a uma variável
amostras = './test-sample/'

# muda o diretório para o diretório do modelo
os.chdir('./modelo-history')
# carrega o modelo salvo
model = tf.keras.models.load_model('modelo.h5')
# retorna ao diretório anterior
os.chdir('..')

# percorre o diretório de amostras
for arquivo in os.walk(amostras):
    for fn in arquivo[2]:
        # carrega a imagem no tamanho correto em que o nodelo foi treinado
        img = tf.keras.utils.load_img(amostras + fn, target_size=(150, 150))
        # converte a imagem em um array
        img_array = tf.keras.utils.img_to_array(img)
        # expande as dimensões da imagem para usar o (r,g,b) para que ela possa ser usada no modelo
        img_array = tf.expand_dims(img_array, 0)
        # faz a predição da imagem
        predictions = model.predict(img_array)
        # retorna o resultado da predição
        score = predictions[0]
        # printa se a imagem é pedra, papel ou tesoura
        print(
            "{} This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(fn, class_names[np.argmax(score)], 100 * np.max(score))
        )