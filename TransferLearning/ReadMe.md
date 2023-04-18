# Flower Recognition Using Transfer Learning
[Suryakiran George](https://www.linkedin.com/in/suryakiran-mg/)


<a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a> 


## Online Demo

[![demo](figs/online_demo.png)]()

## Examples
  |   |   |
:-------------------------:|:-------------------------:
![find wild](figs/examples/wop_2.png) |  ![write story](figs/examples/ad_2.png)
![solve problem](figs/examples/fix_1.png)  |  ![write Poem](figs/examples/rhyme_1.png)

More examples can be found in the [project page](https://minigpt-4.github.io).


## Introduction - Flower Recognition Problem
- Flower recognition in the wildlife has been an area of great interest among biologists
    - To study extinction of different species
    - Educational purposes
    - Useful in gardening, botany research, Ayurveda, farming, Floriculture etc.

- Traditional Approach:
    - Done by a botanist : human limitations
    - Search engines : assists in searching a particular flower but not robust.
    - Google Image search : not always reliable

 - Deep Learning Approach:
    - Rapid advances in Computer Vision & Deep Learning enables building of powerful classification & identification methods
    - Fast & automatic
    - DL Approaches:
        - Build a DL network from scratch : Need large dataset (Available; Eg. ImageNet), Need significant time & computational resources
        - Transfer Learning : Leverage existing models – Use a DL model pre-trained on a large dataset, and adapt it to a smaller dataset of flower images, save significant time & computational resources. The pre-trained model has already learned important features from a large dataset


## Transfer Learning
**Main Idea**
  - Lower-level features(edges, shapes) & higher-level features (textures, patterns) learned by the pre-trained model can be useful for recognizing different types of images, including flowers.
  - Application : 
      - Use the pre-trained model as a feature extractor
      - Remove the final classification layer of the pre-trained model, and replace it with a new classification layer that is specific to our flower dataset
      - Train this new classification layer on the flower dataset, while keeping the pre-trained weights 
      - Fine-tune the pre-trained model to recognize flower-specific features
      - Examples : VGG, ResNet, AlexNet, Inception, MobileNet

**Popular Pre-trained CNNs for Flower Detection**

  - Inception-v3: This is a widely-used convolutional neural network (CNN) architecture for image classification tasks. It has been trained on the ImageNet dataset and has achieved high accuracy in detecting flowers.
  - ResNet-50: This is another popular CNN architecture that has been pre-trained on ImageNet. It has shown good performance in detecting flowers and other objects.
  - VGG-16: This is a deep CNN architecture that has been pretrained on the ImageNet dataset. It has been used for flower detection with good results.
  - MobileNet: This is a lightweight CNN architecture that has been designed to run efficiently on mobile devices. It has been pretrained on the ImageNet dataset and has been used for flower detection tasks.



### Launching Demo Locally


## Acknowledgement


