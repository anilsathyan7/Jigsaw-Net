# Jigsaw-Net

A python script for video composting .

It takes two videos as input and produces a single video composite as output. The person in the first video is segmented out and blended seamlessly into the second video (background) .It uses deeplab v3 for segmentation and a custom network (caffe) for color harmonization.Since the tensorflow model was trained on PASCAL VOC dataset, we can segment out any object belonging to those classes and combine them with the other video to produce a video composite.

Sample input and output videos along with link to pretrained modles are provided in respective folders.

N.B: Currently, only the inference code is provided; please refer the links in the acknowledgement section for training and other implementation details.

## Dependencies

* Python3
* PIL, matplotlib
* Scipy, skimage
* Tensorflow
* Keras 2.2.2

## Prerequisites

* Keras installation (TF backend)  : pip install keras==2.2.2
* GPU with CUDA support

## Demo

### Inputs

Input1:-

![Screenshot](https://drive.google.com/uc?id=1Rb6oXW3KufVApvID9MwxsknpQON2CkQ_)


### Output
![Screenshot](https://drive.google.com/uc?id=1hxKsUA9_oAk41o6YGFuOL9mAgQ5SqxjC)

## Versioning

Version 1.0

## Authors

Anil Sathyan

## Acknowledgments
* "https://averdones.github.io/real-time-semantic-image-segmentation-with-deeplab-in-tensorflow/
* "https://github.com/tensorflow/models/tree/master/research/deeplab"
* "https://github.com/wasidennis/DeepHarmonization"
