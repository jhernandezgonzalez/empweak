import os.path
import pickle

from src.auxiliary import PLDataset, realDataLoader

path = "models_and_data/real/"
datasets = [
    'birdac.mat',
    'lost.mat',
    'MSRCv2.mat',
]
dbfile = open('models_and_data/real_data.pkl', 'wb')

data = []
for file in os.listdir(path):
    if file in datasets:
        print(file)
        data.append( PLDataset(*realDataLoader(path+file), name=file) )

pickle.dump(data, dbfile)

dbfile.close()
