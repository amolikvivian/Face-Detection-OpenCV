# Face-Detection-OpenCV
Face Detection using OpenCV in Python using different methods and models.

## Haar Cascade
The first method utilizes the pretrained Haar Cascade Classifier - frontalface_default to detect faces.
Added a small module to detect eyes from given frame in real time using - haarcascade_eye.

Serves as a pirmitive way to detect faces and eyes, simple to understand and help in starting off with Computer Vision projects.

## Caffe Model
Caffe is a Deep Learning Framework developed by Berkeley AI Research (BAIR). It is widely used for detection in Computer Vision and
provides for a better speed while making predictions. [Caffe Github](https://github.com/BVLC/caffe/).

The current code in the given repository works on the weights - 'res10_300x300_ssd_iter_140000.caffemodel' specifically trained to recognize faces. 

Video Input using imutils - VideoStream.
