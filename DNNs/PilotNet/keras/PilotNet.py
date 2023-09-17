import os
os.environ['HDF5_DISABLE_VERSION_CHECK']='1'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io 
import mat73



#Dataset
input_images = scipy.io.loadmat('DNNs/PilotNet/dataset/PilotNet_images_part1.mat')['images']
label= scipy.io.loadmat('DNNs/PilotNet/dataset/prediction_angels.mat')['prediction_angels'][0:input_images.shape[0],1]/2

#Model
img_input = tf.keras.layers.Input(shape=(66,200,3))
L1 = tf.keras.layers.Conv2D(24, kernel_size=5, strides=(2,2), padding='valid', activation='relu', use_bias=True)(img_input) # output: (31,98,24)
L2 = tf.keras.layers.Conv2D(36, kernel_size=5, strides=(2,2), padding='valid', activation='relu', use_bias=True)(L1)        # output: (14,47,36)
L3 = tf.keras.layers.Conv2D(48, kernel_size=5, strides=(2,2), padding='valid', activation='relu', use_bias=True)(L2)        # output: ( 5,22,48) 
L4 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=(1,1), padding='valid', activation='relu', use_bias=True)(L3)        # output: ( 3,20,64)
L5 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=(1,1), padding='valid', activation='relu', use_bias=True)(L4)        # output: ( 1,18,64)
L5_f = tf.keras.layers.Flatten()(L5)
L6 = tf.keras.layers.Dense(1164, activation='relu', use_bias=True)(L5_f)
L7 = tf.keras.layers.Dense( 100, activation='relu', use_bias=True)(L6)
L8 = tf.keras.layers.Dense(  50, activation='relu', use_bias=True)(L7)
L9 = tf.keras.layers.Dense(  10, activation='relu', use_bias=True)(L8)
L10= tf.keras.layers.Dense(   1, activation='tanh', use_bias=True)(L9)

model = tf.keras.Model(inputs = img_input, outputs =L10, name='PilotNet')
model.compile(loss=tf.keras.losses.MeanSquaredError(), metrics=tf.keras.metrics.MeanSquaredError())
model.summary()

# Set weights
weights = scipy.io.loadmat('DNNs/PilotNet/weights_matlab/PilotNet_weights.mat')
model.layers[ 1].set_weights([weights['W_conv1'],np.squeeze(weights['b_conv1'])])
model.layers[ 2].set_weights([weights['W_conv2'],np.squeeze(weights['b_conv2'])])
model.layers[ 3].set_weights([weights['W_conv3'],np.squeeze(weights['b_conv3'])])
model.layers[ 4].set_weights([weights['W_conv4'],np.squeeze(weights['b_conv4'])])
model.layers[ 5].set_weights([weights['W_conv5'],np.squeeze(weights['b_conv5'])])
model.layers[ 7].set_weights([weights['W_fc1'],np.squeeze(weights['b_fc1'])])
model.layers[ 8].set_weights([weights['W_fc2'],np.squeeze(weights['b_fc2'])])
model.layers[ 9].set_weights([weights['W_fc3'],np.squeeze(weights['b_fc3'])])
model.layers[10].set_weights([weights['W_fc4'],np.squeeze(weights['b_fc4'])])
model.layers[11].set_weights([weights['W_fc5'],weights['b_fc5'][0]])

# Extract weights for sensim
weights=[]
weights.append(model.layers[ 1].get_weights())
weights.append(model.layers[ 2].get_weights())
weights.append(model.layers[ 3].get_weights())
weights.append(model.layers[ 4].get_weights())
weights.append(model.layers[ 5].get_weights())
weights.append(model.layers[ 7].get_weights())
weights.append(model.layers[ 8].get_weights())
weights.append(model.layers[ 9].get_weights())
weights.append(model.layers[10].get_weights())
weights.append(model.layers[11].get_weights())
np.save('./DNNs/'+model.name+'/weights', weights)

# Inference
model.evaluate(input_images, label) # mean_squared_error: 0.0016
prediction=model.predict(input_images, verbose=1)
np.save('./DNNs/'+model.name+'/output_activations.npy', prediction)


# plot the output/GroundTrueth
plt.plot(prediction, linewidth=0.2)
plt.plot(label, linewidth=0.2)
plt.plot(label-prediction[:,0], linewidth=0.3)
plt.legend(['prediction', 'label', 'error'])
plt.show()
plt.savefig('./DNNs/'+model.name+'/output_keras.png', bbox_inches='tight', dpi=2000)
plt.clf()
# NOTE: labels and presictions are based on radians and devided by 2
# To onvert to degress multiply them by 57.2958*2



