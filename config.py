import albumentations as A
import os

WORKING_DIR = './'
WEIGHTS_PATH = './weights/model_weights_final.h5'
IMAGE_DIR = './dataset/imagesTr/'
MASK_DIR = './dataset/masksTr/'
TEST_DIR = './dataset/imagesTs/'

LEARNING_RATE = 1e-4
BATCH_SIZE = 1
DIM = (192, 160, 128)
EPOCHS = 50
NUM_CHANNELS = 4
N_CLASSES = 3
INPUT_SHAPE = (*DIM, NUM_CHANNELS)
STEPS_PER_EPOCH = len(os.listdir(IMAGE_DIR)) // BATCH_SIZE
LOAD_WEIGHTS = True

TRANSFORM = A.Compose([
    A.Normalize(mean=(0., 0., 0., 0.), std=(1., 1., 1., 1.), p=1),
    A.RandomBrightnessContrast(brightness_limit=(0.1), p=0.5),
    A.RandomScale(0.1, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    ])