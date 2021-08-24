import torch


# number of frames in one sliding window
WINDOW_LIMITS = 6
# image size
IMG_SIZE = 64
# number of classes
CLAS = 6
classes = ['normal', 'phone', 'smoking', 'one-hand', 'talking', 'sleeping']

# path of directories
# training data
TRAINING_DATA = r'C:\Users\liewei\Desktop\train\labels.txt'
# others
TEST_DATA = r'C:\Users\liewei\Desktop\test\labels.txt'
ALL_DATA = r'C:\Users\liewei\Desktop\mydata\labels.txt'
TEST_PTH = r'C:\Users\liewei\Desktop\DenseData'

# !!!!!!!!!!!!!!!!!!!!!!! do NOT modify !!!!!!!!!!!!!!!!!!!!!!!!!!!!
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# growth rate of Dense Classifier
GROWTH = 8
# layers in each dense block
DENSE_BNUM = [6, 12, 24, 16]