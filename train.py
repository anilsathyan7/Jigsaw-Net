import numpy as np
import random
from PIL import Image
from imageio import imsave
from skimage.util.shape import view_as_blocks

#Import the data and libraries

import os
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dense, Input, concatenate, Conv2D, Reshape,MaxPooling2D, Lambda,Activation,Conv2DTranspose, UpSampling2D, merge
from keras.layers import UpSampling2D, Conv2DTranspose, BatchNormalization, Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback, ReduceLROnPlateau
from keras.regularizers import l1
from keras.optimizers import SGD, Adam
import keras.backend as K
from keras.backend import tf as ktf
from keras.utils import plot_model
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from random import randint
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from imageio import imsave
from skimage.util.shape import view_as_blocks
from scipy import stats
import matplotlib.pyplot as plt
from random import randint
# %matplotlib inline


# Load the dataset
x_train=np.load("data/shuffle.npy")
y_strain= np.load("data/slabels.npy")
y_rtrain= np.load("data/rlabels.npy")

img_height,img_width, nclasses = 224,224,16
num_images=x_train.shape[0]

# Preprocess the data labels [flow method requires a rank 4 tensor]
y_strain=y_strain.reshape((num_images,224,224,1))
y_rtrain=y_rtrain.reshape((num_images,4,4,1))

# Verify the data types and their shapes
print(x_train.shape, x_train.dtype )
print(y_strain.shape, y_strain.dtype)
print(y_rtrain.shape, y_rtrain.dtype)

# Prepare data generator



image_datagen = ImageDataGenerator()


def generate_data_generator(generator, X, Y1, Y2):
    genX = generator.flow(X, seed=7, batch_size=32)
    genY1 = generator.flow(Y1, seed=7,batch_size=32)
    genY2 = generator.flow(Y2, seed=7,batch_size=32)
    
    while True:
            Xi = genX.next()/255.0
            Yi1 = genY1.next().reshape(-1,50176,1)
            Yi2 = genY2.next().reshape(-1,16,1)
            yield (Xi, {'shuf_op': Yi1, 'rotn_op': Yi2})

train_gen = generate_data_generator(image_datagen,
                                    x_train[0:4000,...],
                                    y_strain[0:4000,...],
                                    y_rtrain[0:4000,...])

val_gen = generate_data_generator(image_datagen,
                                  x_train[4000:,...],
                                  y_strain[4000:,...],
                                  y_rtrain[4000:,...])


# Convoltuion blocks
def conv_block(tensor, nfilters, size=3, padding='same', kernel_initializer = 'he_normal'):
    x = Conv2D(filters=nfilters, kernel_size=(size,size) , padding=padding, kernel_initializer = 'he_normal')(tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x
  
def deconv_block(tensor, residual, nfilters, size=3, padding='same', kernel_initializer = 'he_normal'):
    y = UpSampling2D(size = (2,2))(tensor)
    y = Conv2D(filters=nfilters, kernel_size=(size,size), activation = 'relu', padding = 'same', kernel_initializer = kernel_initializer)(y)
    y = concatenate([y,residual], axis = 3)
    y = conv_block(y, nfilters)
    
    return y

# Define the network architecture
def get_jnet():
    inputs = Input((224,224,3))
    
    #Contraction path
    conv1= conv_block(inputs, 16)
    pool1 = MaxPooling2D(pool_size=2)(conv1)

    conv2 = conv_block(pool1, 32)
    pool2 = MaxPooling2D(pool_size=2)(conv2)
   

    conv3 = conv_block(pool2, 64)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = conv_block(pool3, 128)
    pool4 = MaxPooling2D(pool_size=2)(conv4)
   
    conv5 = conv_block(pool4, 256)
   
    
    #Expansion path
    up6 = deconv_block(conv5, conv4, 128)
    up7 = deconv_block(up6, conv3, 64)
    up8 = deconv_block(up7, conv2, 32)
    up9= deconv_block(up8, conv1, 16)
    

    conv10 = Conv2D(16, kernel_size=(1, 1))(up9)
    out1 = BatchNormalization()(conv10)
    out1 = Reshape((img_height*img_width, nclasses), input_shape=(img_height, img_width, nclasses))(out1)
    out1 = Activation('softmax', name="shuf_op")(out1)
     
    pool_fin = MaxPooling2D(pool_size=2)(up9)
    conv11 = Conv2D(4, kernel_size=(28, 28), strides=(28, 28))(pool_fin)
    out2 = BatchNormalization()(conv11)
    out2 = Reshape((-1, 4), input_shape=(4, 4, 4))(out2)
    out2 = Activation('softmax', name= "rotn_op")(out2)
    
    losses = { "shuf_op": "sparse_categorical_crossentropy",
	             "rotn_op": "sparse_categorical_crossentropy" }
             
    lossWeights = {"shuf_op": 1.0, "rotn_op": 1.0}

    model = Model(inputs=[inputs], outputs=[out1, out2])
    model.compile(optimizer = Adam(lr = 1e-4), loss=losses, loss_weights=lossWeights, metrics = ['sparse_categorical_accuracy'])

    return model

model=get_jnet()

# Save the model summary
model.summary()
plot_model(model, to_file='scramble.png')



# Save the checkpoints and monitor training progress
filepath="checkpoints/scramble_model-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only=False , save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=15, min_lr=0.00001, verbose=1)
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=True)

callbacks_list = [tensorboard, checkpoint,reduce_lr]

# Train the network

model.fit_generator(
    train_gen,
    epochs=300,
    steps_per_epoch=125,
    validation_data=val_gen,
    validation_steps=31,
    callbacks=callbacks_list)
