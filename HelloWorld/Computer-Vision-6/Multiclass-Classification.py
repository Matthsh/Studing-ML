# wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip -O /tmp/rps.zip

# importa a biblioteca os para manipular arquivos
import zipfile
# importa a biblioteca ImageDataGenerator para gerar dados de imagens
from keras.preprocessing.image import ImageDataGenerator


# atribui o local do zip a uma variável
local_zip = '/Computer-Vision-6/rps.zip'


# abre o arquivo em modo de leitura(read)
zip_ref = zipfile.ZipFile(local_zip, 'r')
# extrai o arquivo zipado
zip_ref.extractall('/Computer-Vision-6')
# fecha o arquivo
zip_ref.close()


# define o nome do diretório base
TRAINING_DIR = "/Computer-Vision-6/rps/"
# define o nome do diretório de treino e alguns parametros de formatação para as imagens
training_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest'
)


# gera os dados de treino
train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150,150),
    class_mode='categorical'
)