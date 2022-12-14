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
Pix2Pix GAN’s have some conditional settings and they learn the image-to-image mapping under this condition. Whereas, basic GAN’s generate images from a random distribution vector with no condition applied.</br>
The architecture of Pix2Pix will contain: 
  * A generator with a U-Net-based architecture. 
  * A discriminator represented by a convolutional PatchGAN classifier.

### Pix2Pix Generator
The generator of pix2pix CGAN is a modified U-Net.
* A U-Net consists of an encoder (downsampler) and decoder (upsampler).</br>
<img src= https://github.com/swethareddy23/Generating-maps-from-satellite-images/blob/main/Generator.png width='400' height='250' /></br>
#### Generator Network:
<img src= https://github.com/swethareddy23/Generating-maps-from-satellite-images/blob/main/Generator_1.png width='400' height='250' />


### Pix2Pix Discriminator
* The discriminator in the pix2pix cGAN is a Convolutional PatchGAN classifier—it tries to classify if each image patch is real or not real.</br>
<img src= https://github.com/swethareddy23/Generating-maps-from-satellite-images/blob/main/Discriminator.png width='400' height='250' /></br>
#### Discrimintaor Network:
<img src= https://github.com/swethareddy23/Generating-maps-from-satellite-images/blob/main/Discriminator_1.png width='400' height='250' /></br>

## Generator Loss:
The formula to calculate the total generator loss is 
 * BCE loss+ LAMBDA * ∑i=1 to n  |generated_output - real_output|.</br>
<img src= https://github.com/swethareddy23/Generating-maps-from-satellite-images/blob/main/Generator_Loss.png width='250' height='150' /></br>

## Discriminator loss:
The discriminator loss function takes 2 inputs: real images and generated images.
*	Real loss is a Binary cross-entropy loss of the real images and an array of ones (since these are the real images).
*	Generated loss is a Binary cross-entropy loss of the generated images and an array of zeros (since these are the fake images).
*	The total loss is the sum of real loss and generated loss.</br>
<img src= https://github.com/swethareddy23/Generating-maps-from-satellite-images/blob/main/Discriminator_loss.png width='400' height='300' /></br>

## Training Results:
* Epoch=0 </br>
<img src= https://github.com/swethareddy23/Generating-maps-from-satellite-images/blob/main/Training/Satellite_image_epoch0.png width='650' height='200' /></br>
<img src= https://github.com/swethareddy23/Generating-maps-from-satellite-images/blob/main/Training/Ground_truth_epoch0.png width='650' height='200' /></br>
<img src= https://github.com/swethareddy23/Generating-maps-from-satellite-images/blob/main/Training/resultmap_epoch0.png width='650' height='200' /></br>


* Epoch=10 </br>
<img src= https://github.com/swethareddy23/Generating-maps-from-satellite-images/blob/main/Training/Satellite_image_epoch10.png width='650' height='200' /></br>
<img src= https://github.com/swethareddy23/Generating-maps-from-satellite-images/blob/main/Training/Ground_truth_epoch10.png width='650' height='200' /></br>
<img src= https://github.com/swethareddy23/Generating-maps-from-satellite-images/blob/main/Training/resultmap_epoch10.png width='650' height='200' /></br>

* Epoch=20 </br>
<img src= https://github.com/swethareddy23/Generating-maps-from-satellite-images/blob/main/Training/Satellite_image_epoch20.png width='650' height='200' /></br>
<img src= https://github.com/swethareddy23/Generating-maps-from-satellite-images/blob/main/Training/Ground_truth_epoch20.png width='650' height='200' /></br>
<img src= https://github.com/swethareddy23/Generating-maps-from-satellite-images/blob/main/Training/resultmap_epoch20.png width='650' height='200' /></br>

## Testing Results:
<img src= https://github.com/swethareddy23/Generating-maps-from-satellite-images/blob/main/test.png width='650' height='200' /></br>
