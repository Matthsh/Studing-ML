# Code to get the training data and extract it into the
# appropriately named subdirectories

# importa as bibliotécas necessárias
import urllib.request
import zipfile

# salva o url da pasta aonde as imagens estão na web
url = "http://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip"

# salva o nome da pasta apenas
file_name = "horse-or-human.zip"
# salva o nome do diretório aonde as imagens serão extraidas
training_dir = 'horse-or-human/training/'

# faz o download do arquivo através do url especificado
urllib.request.urlretrieve(url, file_name)

# salva uma classe zipfile com o arquivo baixado
zip_ref = zipfile.ZipFile(file_name, 'r')
# usa uma função da classe zipfile para extrai todos os arquivos da pasta zipada dentro de uma pasta comum 
zip_ref.extractall(training_dir)
# fecha a classe de zipfile
zip_ref.close()