import scipy
import sklearn
from sklearn.feature_extraction import image
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
images = loadmat('PilotNet_images_part1.mat')['images'][1:250]
savemat('pilotnet_0_250.mat', mdict={'images': images[0]})
