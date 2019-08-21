import numpy as np
import random
from PIL import Image
from imageio import imsave
from skimage.util.shape import view_as_blocks


#Preprare labels
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

#Prepare images
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

# Load the source image and generate it's label
im=np.array(Image.open('data/angrybird.jpeg').convert('RGB'))
lb=label(im,4)
 

# Generate the shuffled dataset
images=[]
labels_shuf=[]
labels_rotn=[]

for i in range(0,5000):
 img,lbl_shuf,lbl_rotn=shuffle(im,lb,4,rotate=True)
 images.append(img)
 labels_shuf.append(lbl_shuf)
 labels_rotn.append(lbl_rotn)

imgset=np.array(images)
lblshufset=np.array(labels_shuf)
lblrotnset=np.array(labels_rotn)

# Verify the shapes
print (imgset.shape)
print (lblshufset.shape)
print (lblrotnset.shape)

# Save the dataset as numpy arrays
np.save("data/shuffle.npy",imgset)
np.save("data/slabels.npy",lblshufset)
np.save("data/rlabels.npy",lblrotnset)
