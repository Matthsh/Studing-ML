# wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip -O /home/matthsh/Documentos/Studing-ML/HelloWorld/Computer-Vision-6/rps.zip

#importar as bibliotecas necessárias
import urllib.request
import zipfile

# define o url do arquivo zipado
url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip'
# define o nome do arquivo zipado
zip_file = 'rps.zip'

# faz o download do arquivo zipado
urllib.request.urlretrieve(url, zip_file)


# abre o arquivo em modo de leitura(read)
zip_ref = zipfile.ZipFile(zip_file, 'r')
# extrai o arquivo zipado
zip_ref.extractall()
# fecha o arquivo
zip_ref.close()