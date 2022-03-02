# U-Net implementation in Jax using Haiku neural network library

U-Net is a special case of classic encoder-decoder model with skip connections between encoder-decoder layers and mainly used for image segmentation tasks. In this example, Oxford Pets dataset used to segmentate pets without any further classification applied on them. This work directly inspired by François Chollet's [Image Segmentation with a U-Net-like Architecture](https://keras.io/examples/vision/oxford_pets_image_segmentation)


![ed-unet](https://user-images.githubusercontent.com/88535469/151528034-c41a7ea6-2f48-49ef-9e2b-1a9dbcf9101c.png)


In most cases, skipping layer used between encoder-decoder layers. In this example it is used between layers inside blocks and this probably the reason of why it is named "U-Net-like Architecture", but the example still captures the all essence required to create a more complex U-Nets. Please check [Tensorflow Pix2Pix](https://www.tensorflow.org/tutorials/generative/pix2pix) example if still want to see simple implementation of skipping layers between encoder and decoder.



