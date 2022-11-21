import os


pasta = './content'

for arquivo in os.walk(pasta):
    for fn in arquivo[2]:

        print(fn)