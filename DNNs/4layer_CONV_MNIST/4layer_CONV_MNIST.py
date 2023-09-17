import os
os.environ['HDF5_DISABLE_VERSION_CHECK']='1'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images / 255
test_images = test_images / 255
train_images=np.expand_dims(train_images, axis=3)
test_images =np.expand_dims(test_images, axis=3)

#Model
img_input = tf.keras.layers.Input(shape=(28,28,1))
L1 = tf.keras.layers.Conv2D(8, kernel_size=3, strides=(2,2), padding='valid', activation='relu', use_bias=True)(img_input) # output: (13,13,8)
L2 = tf.keras.layers.Conv2D(16, kernel_size=3, strides=(2,2), padding='valid', activation='relu', use_bias=True)(L1) # output: (6,6,16)
L3 = tf.keras.layers.Conv2D(32, kernel_size=3, strides=(2,2), padding='valid', activation='relu', use_bias=True)(L2) # output: (2,2,32) 
L3_f = tf.keras.layers.Flatten()(L3)
Predictions = tf.keras.layers.Dense(10, use_bias=True)(L3_f)
model = tf.keras.Model(inputs = img_input, outputs = Predictions, name='4Layer_CONV_MNIST')
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
model.summary()

#Training
training = 0
if training:
    model.fit(train_images, train_labels, epochs=50, batch_size=128, validation_data=(test_images, test_labels))
    model.evaluate(test_images, test_labels)
    model.save_weights('./DNNs/'+model.name+'/weights.h5')

#Extract wegihts
extract_weights = 0
weights=[]
model.load_weights('./DNNs/'+model.name+'/weights.h5')
if extract_weights:
    model.evaluate(test_images, test_labels)

    weights.append(model.layers[1].get_weights()) # conv1
    weights.append(model.layers[2].get_weights()) # conv2
    weights.append(model.layers[3].get_weights()) # conv3
    weights.append(model.layers[5].get_weights()) # dense1

    np.save('./DNNs/'+model.name+'/weights', weights)

#Extract some of the outputs
num_extract_images = 3

# intermediate layer results
model1 = tf.keras.Model(inputs = img_input, outputs = L1)
prediction=model1.predict(test_images[0:num_extract_images])
np.save('./DNNs/'+model.name+'/L1_activations', prediction)

model2 = tf.keras.Model(inputs = img_input, outputs = L2)
prediction=model2.predict(test_images[0:num_extract_images])
np.save('./DNNs/'+model.name+'/L2_activations', prediction)

model3 = tf.keras.Model(inputs = img_input, outputs = L3)
prediction=model3.predict(test_images[0:num_extract_images])
np.save('./DNNs/'+model.name+'/L3_activations', prediction)

# outputs
prediction=model.predict(test_images[0:num_extract_images])
np.save('./DNNs/'+model.name+'/output_activations', prediction)



###### predict with synthetic input ########
# test_images_syn = np.zeros([1,28,28,1])
# test_images_syn[0,0,0,0] = 1
# test_images_syn[0,5,5,0] = 1
# test_images_syn[0,10,10,0] = 1
# test_images_syn[0,20,20,0] = 1
# test_images_syn[0,26,26,0] = 1
# model1 = tf.keras.Model(inputs = img_input, outputs = L1)
# prediction=model1.predict(test_images_syn)
# np.save('./DNNs/'+model.name+'/L1_activations_syn', prediction)    
# plt.imshow(prediction[0,:,:,0])
# plt.show()         