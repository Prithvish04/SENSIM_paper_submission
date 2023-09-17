import tensorflow as tf
import numpy as np
import tensorflow_model_optimization as tfmot


#Dataset
binarize_input = 1
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
if binarize_input:
    train_images = (train_images>64).astype('float32') #binary
    test_images  = (test_images>64).astype('float32') #binary
else:
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
model = tf.keras.Model(inputs = img_input, outputs = Predictions, name='4layer_CONV_MNIST')
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
model.load_weights('./DNNs/'+model.name+'/weights.h5')
model.summary()


# quantized model
num_bits = 6
quantized_layers = [1,2,3,5]
for i in quantized_layers:
    W_max = np.max([np.max(np.abs(model.layers[i].get_weights()[0])),np.max(np.abs(model.layers[i].get_weights()[1]))])
    W = np.ceil(model.layers[i].get_weights()[0]*(2**num_bits-1)/W_max)
    B = np.ceil(model.layers[i].get_weights()[1]*(2**num_bits-1)/W_max)
    model.layers[i].set_weights([W,B])
model.evaluate(test_images, test_labels, verbose=1)

#Extract wegihts
extract_weights = 1
weights=[]
if extract_weights:
    weights.append(model.layers[1].get_weights()) # conv1
    weights.append(model.layers[2].get_weights()) # conv2
    weights.append(model.layers[3].get_weights()) # conv3
    weights.append(model.layers[5].get_weights()) # dense1
    np.save('./DNNs/'+model.name+'/quantized_weights_'+str(num_bits)+'b', weights)

#Extract some of the outputs
num_extract_images = 3

# intermediate layer results
model1 = tf.keras.Model(inputs = img_input, outputs = L1)
prediction=model1.predict(test_images[0:num_extract_images])
np.save('./DNNs/'+model.name+'/L1_activations_quantized', prediction)
print("Layer1 max value:", np.max(abs(prediction)))

model2 = tf.keras.Model(inputs = img_input, outputs = L2)
prediction=model2.predict(test_images[0:num_extract_images])
np.save('./DNNs/'+model.name+'/L2_activations_quantized', prediction)
print("Layer2 max value:", np.max(abs(prediction)))

model3 = tf.keras.Model(inputs = img_input, outputs = L3)
prediction=model3.predict(test_images[0:num_extract_images])
np.save('./DNNs/'+model.name+'/L3_activations_quantized', prediction)
print("Layer3 max value:", np.max(abs(prediction)))

# outputs
prediction=model.predict(test_images[0:num_extract_images])
np.save('./DNNs/'+model.name+'/output_activations_quantized', prediction)  
print("Layer4 max value:", np.max(abs(prediction)))  
print("Layer4 max bitwidth:", np.log2(np.max(abs(prediction))+1))  