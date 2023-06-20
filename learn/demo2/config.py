# from model.net18_2 import SiameseNetwork
# from model.net50_1 import SiameseNetwork
from model.net50_3 import SiameseNetwork

input_shape     = [64, 64]
batch_size          = 64
save_dir = 'logs'

def getModel():
    return SiameseNetwork()