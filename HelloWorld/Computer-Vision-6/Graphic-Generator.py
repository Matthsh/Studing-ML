# script para gerar gráficos de acurácia e perda a partir do arquivo de histórico de treinamento salvo

# importa a biblioteca matplotlib
import matplotlib.pyplot as plt
import pickle

# define o nome do arquivo de histórico de treinamento
history_file = '/home/matthsh/Documentos/Studing-ML/HelloWorld/Computer-Vision-6/modelo-history/history.pickle'

# carrega o arquivo de histórico de treinamento
with open(history_file, 'rb') as file_pi:
    history = pickle.load(file_pi)

# gráfico de acurácia
acc = history['accuracy']

# define o tamanho do gráfico
plt.figure(figsize=(8, 8))
# define o número de linhas e colunas do gráfico
plt.subplot(2, 1, 1)
# plota o gráfico de acurácia
plt.plot(acc, label='Training Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

# mostra na tela o gráfico
plt.show()