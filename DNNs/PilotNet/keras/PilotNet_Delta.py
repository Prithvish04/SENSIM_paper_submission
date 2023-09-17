from os import name
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import mat73
from DeltaActivationLayer import DAL


#Dataset
input_images = scipy.io.loadmat('DNNs/PilotNet/dataset/PilotNet_images_part1.mat')['images']
label = scipy.io.loadmat('DNNs/PilotNet/dataset/prediction_angels.mat')['prediction_angels'][1:len(input_images),1]/2
PilotNet_output = np.load('DNNs/PilotNet/output_activations_AllFrames.npy')[1:len(input_images)][:,0]

input_sequences = np.zeros([input_images.shape[0]-1, 2, input_images.shape[1], input_images.shape[2], input_images.shape[3]])
for i in range(len(input_images)-1):
    input_sequences[i,:,:,:,:]  = input_images[i:i+2, :,:,:]

# parameter adjustment, with threshold=0 --> mean_squared_error: 4.6153e-16 - Operation_Density: 0.6477
sp_rate = 1e-4
thr_init = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])*1e-1
thr_trainable = True
#Model
img_input = tf.keras.layers.Input(shape=(2, 66,200,3), name='input_0')
inp_delta, S0, N0 = DAL(sp_rate=sp_rate, thr_init=thr_init[0], name='DAL0', n_outputs=24*5*5/4, show_metric=False, thr_trainable=thr_trainable)(img_input)

L1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(24, kernel_size=5, strides=(2,2), padding='valid', activation='relu', use_bias=True), name='conv1')(inp_delta) # output: (31,98,24)
L1_delta, S1, N1 = DAL(sp_rate=sp_rate, thr_init=thr_init[0], name='DAL1', n_outputs=36*5*5/4, show_metric=False, thr_trainable=thr_trainable)(L1)

L2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(36, kernel_size=5, strides=(2,2), padding='valid', activation='relu', use_bias=True), name='conv2')(L1_delta)  # output: (14,47,36)
L2_delta, S2, N2 = DAL(sp_rate=sp_rate, thr_init=thr_init[1], name='DAL2', n_outputs=48*5*5/4, show_metric=False, thr_trainable=thr_trainable)(L2)

L3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(48, kernel_size=5, strides=(2,2), padding='valid', activation='relu', use_bias=True), name='conv3')(L2_delta)  # output: ( 5,22,48) 
L3_delta, S3, N3 = DAL(sp_rate=sp_rate, thr_init=thr_init[2], name='DAL3', n_outputs=64*3*3, show_metric=False, thr_trainable=thr_trainable)(L3)

L4 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, kernel_size=3, strides=(1,1), padding='valid', activation='relu', use_bias=True), name='conv4')(L3_delta)  # output: ( 3,20,64)
L4_delta, S4, N4 = DAL(sp_rate=sp_rate, thr_init=thr_init[3], name='DAL4', n_outputs=64*3*3, show_metric=False, thr_trainable=thr_trainable)(L4)

L5 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, kernel_size=3, strides=(1,1), padding='valid', activation='relu', use_bias=True), name='conv5')(L4_delta)  # output: ( 1,18,64)
L5_delta, S5, N5 = DAL(sp_rate=sp_rate, thr_init=thr_init[4], name='DAL5', n_outputs=1164, show_metric=False, thr_trainable=thr_trainable)(L5)

L5_f = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(L5_delta)
L6 = tf.keras.layers.Dense(1164, activation='relu', use_bias=True, name='fc1')(L5_f)
L6_delta, S6, N6 = DAL(sp_rate=sp_rate, thr_init=thr_init[5], name='DAL6', n_outputs=100, show_metric=False, thr_trainable=thr_trainable)(L6)

L7 = tf.keras.layers.Dense( 100, activation='relu', use_bias=True, name='fc2')(L6_delta)
L7_delta, S7, N7 = DAL(sp_rate=sp_rate, thr_init=thr_init[6], name='DAL7', n_outputs=50, show_metric=False, thr_trainable=thr_trainable)(L7)

L8 = tf.keras.layers.Dense(  50, activation='relu', use_bias=True, name='fc3')(L7_delta)
L8_delta, S8, N8 = DAL(sp_rate=sp_rate, thr_init=thr_init[7], name='DAL8', n_outputs=10, show_metric=False, thr_trainable=thr_trainable)(L8)

L9 = tf.keras.layers.Dense(  10, activation='relu', use_bias=True, name='fc4')(L8_delta)
L9_delta, S9, N9 = DAL(sp_rate=sp_rate, thr_init=thr_init[8], name='DAL9', n_outputs=1, show_metric=False, thr_trainable=thr_trainable)(L9)

L10= tf.keras.layers.Dense(   1, activation='tanh', use_bias=True, name='output')(L9_delta)

model = tf.keras.Model(inputs = img_input, outputs =L10[:,1,0], name='PilotNet_Delta')

Operations  = S0*24*5*5/4 + S1*36*5*5/4 + S2*48*5*5/4 + S3*64*3*3 + S4*64*3*3 + S5*1164 + S6*100 + S7*50 + S8*10 + S9*1
Synapses    = N0*24*5*5/4 + N1*36*5*5/4 + N2*48*5*5/4 + N3*64*3*3 + N4*64*3*3 + N5*1164 + N6*100 + N7*50 + N8*10 + N9*1
model.add_metric(Operations/Synapses,  name='Operation_Density',    aggregation='mean') #moving average number operation density

model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.MeanSquaredError(), metrics=tf.keras.metrics.MeanSquaredError())
model.summary()

# Set weights
weights = scipy.io.loadmat('DNNs/PilotNet/weights_matlab/PilotNet_weights.mat')
model.layers[ 2].set_weights([weights['W_conv1'],np.squeeze(weights['b_conv1'])])
model.layers[ 4].set_weights([weights['W_conv2'],np.squeeze(weights['b_conv2'])])
model.layers[ 6].set_weights([weights['W_conv3'],np.squeeze(weights['b_conv3'])])
model.layers[ 8].set_weights([weights['W_conv4'],np.squeeze(weights['b_conv4'])])
model.layers[10].set_weights([weights['W_conv5'],np.squeeze(weights['b_conv5'])])
model.layers[13].set_weights([weights['W_fc1'],np.squeeze(weights['b_fc1'])])
model.layers[15].set_weights([weights['W_fc2'],np.squeeze(weights['b_fc2'])])
model.layers[17].set_weights([weights['W_fc3'],np.squeeze(weights['b_fc3'])])
model.layers[19].set_weights([weights['W_fc4'],np.squeeze(weights['b_fc4'])])
model.layers[21].set_weights([weights['W_fc5'],weights['b_fc5'][0]])

# Freeze the weights of the original network
model.layers[ 2].trainable = False
model.layers[ 4].trainable = False
model.layers[ 6].trainable = False
model.layers[ 8].trainable = False
model.layers[10].trainable = False
model.layers[13].trainable = False
model.layers[15].trainable = False
model.layers[17].trainable = False
model.layers[19].trainable = False
model.layers[21].trainable = False


# Training for Treshold optimization 
training = 1
if training:
    model.fit(input_sequences, PilotNet_output, batch_size=32, epochs=10)
    model.save_weights('./DNNs/PilotNet/weights_delta.h5')
else:
    model.load_weights('./DNNs/PilotNet/weights_delta.h5')
    model.evaluate(input_sequences, PilotNet_output)
# Extract weights for sensim
extract_weights = 1
if extract_weights:
    weights=[]
    weights.append(model.layers[ 2].get_weights())
    weights.append(model.layers[ 4].get_weights())
    weights.append(model.layers[ 6].get_weights())
    weights.append(model.layers[ 8].get_weights())
    weights.append(model.layers[10].get_weights())
    weights.append(model.layers[13].get_weights())
    weights.append(model.layers[15].get_weights())
    weights.append(model.layers[17].get_weights())
    weights.append(model.layers[19].get_weights())
    weights.append(model.layers[21].get_weights())
    np.save('./DNNs/PilotNet/weights_delta', weights)

    # Extract Thresholds for sensim
    weights=[]
    weights.append(model.layers[ 1].get_weights()[0][0])
    weights.append(model.layers[ 3].get_weights()[0][0])
    weights.append(model.layers[ 5].get_weights()[0][0])
    weights.append(model.layers[ 7].get_weights()[0][0])
    weights.append(model.layers[ 9].get_weights()[0][0])
    weights.append(model.layers[11].get_weights()[0][0])
    weights.append(model.layers[14].get_weights()[0][0])
    weights.append(model.layers[16].get_weights()[0][0])
    weights.append(model.layers[18].get_weights()[0][0])
    weights.append(model.layers[20].get_weights()[0][0])
    np.save('./DNNs/PilotNet/thresholds_delta', weights)

    # Inference
    prediction=model.predict(input_sequences, verbose=1)
    np.save('./DNNs/PilotNet/output_delta.npy', prediction)


# plot the output/GroundTrueth
plt.plot(prediction[0:1000], linewidth=0.4)
plt.plot(PilotNet_output[0:1000], linewidth=0.4)
plt.legend(['Delta_output', 'PolotNet_output'])
plt.show()
plt.savefig('./DNNs/PilotNet/output_delta_keras.png', bbox_inches='tight', dpi=1200)
# NOTE: labels and presictions are based on radians and devided by 2
# To onvert to degress multiply them by (180/np.pi)*2



