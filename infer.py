
import numpy as np
import random
from PIL import Image
from scipy import stats
from imageio import imsave
import matplotlib.pyplot as plt
from skimage.util.shape import view_as_blocks


import keras
from keras.models import load_model
from keras.models import Model



#Prepare test image
def shuffle(im, lb, num, rotate= False):
  map = {}
  rows=cols=num
  blk_size=im.shape[0]//rows
   
  img_blks=view_as_blocks(im,block_shape=(blk_size,blk_size,3)).reshape((-1,blk_size,blk_size,3))
  lbl_blks=view_as_blocks(lb,block_shape=(blk_size,blk_size)).reshape((-1,blk_size,blk_size))
  
   
  img_shuff=np.zeros((im.shape[0],im.shape[1],3),dtype=np.uint8)
  lbl_shuff=np.zeros((lb.shape[0],lb.shape[1]),dtype=np.uint8)
  lbl_rotn=np.zeros((lb.shape[0]//blk_size,lb.shape[1]//blk_size),dtype=np.uint8)
   
  a=np.arange(rows*rows, dtype=np.uint8)
  b=np.random.permutation(a)
  
  map = {k:v for k,v in zip(a,b)}
  print ("Key Map:-\n" + str(map))
  
  for i in range(0,rows):
    for j in range(0,cols):
     x,y = i*blk_size, j*blk_size
     if(rotate):
      rot_val=random.randrange(0,4)
      lbl_rotn[i,j]=rot_val
      img_shuff[x:x+blk_size, y:y+blk_size] = np.rot90(img_blks[map[i*rows + j]],rot_val)
      lbl_shuff[x:x+blk_size, y:y+blk_size] = lbl_blks[map[i*rows + j]]
     else:
      img_shuff[x:x+blk_size, y:y+blk_size] = img_blks[map[i*rows + j]]
      lbl_shuff[x:x+blk_size, y:y+blk_size] = lbl_blks[map[i*rows + j]]  
  return img_shuff,lbl_shuff,lbl_rotn


#Preprare test label
def label(im, num):
 
  rows=cols=num
  blk_size=im.shape[0]//rows
  img_lab=np.zeros((im.shape[0], im.shape[1]),dtype=np.uint8)
  print (img_lab.shape)
  
  for i in range(0,rows):
     for j in range(0,cols):
      x,y = i*blk_size, j*blk_size
      img_lab[x:x+blk_size, y:y+blk_size] = np.full((blk_size, blk_size),i*rows+j)
                                             
  return img_lab


# Post-process function

'''
Function to decode image
from model output
Args:-
im - input image
skey - shuffle key
rkey - rotation key
num - number of blocks in a row/column
'''
def unscramble(im, skey,rkey, num):
 
  map={}
  rows=cols=num
  blk_size=im.shape[0]//rows
  img_blks=view_as_blocks(im,block_shape=(blk_size,blk_size,3)).reshape((-1,blk_size,blk_size,3))
  rkey=rkey.flatten()
  img_shuff=np.zeros((im.shape[0], im.shape[0],3),dtype=np.uint8)
  
  for i in range(0,rows):
     for j in range(0,cols):
      x,y = i*blk_size, j*blk_size
      map[i*rows+j] = stats.mode(skey[x:x+blk_size, y:y+blk_size] ,axis=None)[0][0]

  inv_map = {v: k for k, v in map.items()}
  
  for i in range(0,rows):
    for j in range(0,cols):
     x,y = i*blk_size, j*blk_size
     img_shuff[x:x+blk_size, y:y+blk_size] = np.rot90(img_blks[inv_map[i*rows + j]] ,(rkey[inv_map[i*rows + j]] *3)%4 )
  
  
  return img_shuff

# Load the source image and generate it's label
im=np.array(Image.open('data/angrybird.jpeg').convert('RGB'))
lb=label(im,4)

# Generate a random shuffle-input (4 rows & 4 columns)
input_image,slabel,rlabel=shuffle(im, lb, 4, rotate= True)

#Normalize input
input_image=input_image/255.0

#Load the model
model=load_model('data/rscramble_model-300-0.00.hdf5',compile=False)

# Predict output
sout,rout=model.predict(input_image.reshape(1,224,224,3))

print(sout.shape)
print(rout.shape)

#Extract sparse labels
sout=np.uint8(np.argmax(sout,axis=2)).reshape((224,224))
rout=np.uint8(np.argmax(rout,axis=2)).reshape((4,4))

print(sout.shape)
print(rout.shape)

# Plot the outputs

# Input
plt.figure("Input image")
plt.imshow(input_image)

# Decode the output
result=unscramble(np.uint8(input_image*255.0),sout,rout,4)

# Output
plt.figure("Output image")
plt.imshow(result)


# Original image
img=np.array(Image.open('data/angrybird.jpeg'))
plt.figure("Original Image")
plt.imshow(img)

plt.show()

