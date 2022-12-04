# Generating-maps-from-satellite-images
Image to Image translation is employed to convert satellite images to the corresponding map images

## Dataset
>[Link to Dataset](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz)</br>
* The dataset contains satellite images and their corresponding images of maps stacked side by side. </br>
* The number of images in data-set (train+validation) : 2194 </br>
* The Size of each combined Image (the satellite and the map image) in data-set : 1200\*600 
(so that means each image (map or satellite iamge) is of size 600\*600)

## Model Architecture
Model Used: Pix2Pix GAN </br>
The models aim at finding the patterns between the input and output image. Image to image translation is employed to convert satellite images to the corresponding maps.</br>
Pix2Pix GAN’s have some conditional settings and they learn the image-to-image mapping under this condition. Whereas, basic GAN’s generate images from a random distribution vector with no condition applied.
The architecture of pix2pix will contain: 
* A generator with a U-Net-based architecture. 
* A discriminator represented by a convolutional PatchGAN classifier.

