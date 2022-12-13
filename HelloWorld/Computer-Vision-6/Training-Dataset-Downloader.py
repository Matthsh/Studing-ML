# wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip -O /home/matthsh/Documentos/Studing-ML/HelloWorld/Computer-Vision-6/rps.zip

#importar as bibliotecas necessárias
import urllib.request
import zipfile
import os

# define o url do arquivo zipado
url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip"
# define o nome do arquivo zipado
file_name = "rps.zip"

if not os.path.exists("training-dataset"):
    os.mkdir("training-dataset")

training_dir = "training-dataset/"

# mudar a pasta do arquivo zipado
os.chdir(training_dir)
# faz o download do arquivo zipado dentro de training_dir
urllib.request.urlretrieve(url, file_name)


# abre o arquivo em modo de leitura(read)
zip_ref = zipfile.ZipFile(file_name, 'r')
# extrai o arquivo zipado
zip_ref.extractall(training_dir)
# fecha o arquivo
zip_ref.close()

# executa o script Multiclass-Classification.py
os.system('python3 /home/matthsh/Documentos/Studing-ML/HelloWorld/Computer-Vision-6/Multiclass-Classification.py')