# Flower Recognition Using Transfer Learning
[Suryakiran G.](https://www.linkedin.com/in/suryakiran-mg/)




## Online Demo

[~ Demo](https://meetsuki-gradio-flower-recognizer.hf.space)

## Examples


More examples can be found in the [project page](https://meetsuki-gradio-flower-recognizer.hf.space)


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



## Abstract

This project focuses on developing a flower recognition system using transfer learning and several pre-trained neural networks. The objective of the project is to achieve high accuracy in identifying different types of flowers from images, by leveraging the power of pre-trained models that have already been trained on large datasets.

Several pre-trained models, including VGG19, ResNet50, MobileNetV2 and a custom built CNN, are used to extract high-level features from flower images. The extracted features will then be fed into a fully connected neural network that will be trained on a small dataset of flower images specific to the project.

The project aims to evaluate the performance of each pre-trained model and compare the results to select the most suitable model for the given dataset. The models will be trained and tested on a flower dataset containing several hundred images of various flower species. The performance metrics will be evaluated based on accuracy, precision, recall, and F1-score.

The developed flower recognition models can be used in various applications such as horticulture, gardening, and agriculture. It will enable automatic identification and classification of flowers in real-time, significantly reducing the manual effort and time required in these applications.



## Method

There are several pre-trained neural networks that can be used for flower image classification tasks. Some of the most popular ones include:

**MobileNet** - MobileNet is a lightweight convolutional neural network designed for mobile and embedded vision applications. Its small size makes it ideal for resource-constrained environments. MobileNetV2 and MobileNetV3 have been used for flower image classification.

**ResNet** (Residual Network) - ResNet is a deep neural network architecture that has been very successful in image classification tasks. The ResNet50 and ResNet101 models have been used for flower image classification with good results.

**VGG** (Visual Geometry Group) - VGG models are known for their simplicity and have achieved state-of-the-art results in various image classification tasks. The VGG16 and VGG19 models are commonly used for flower image classification.
Pre-trained models can be fine-tuned on flower image datasets to improve their accuracy and achieve good results in flower classification tasks. A simple custom built CNN is also used in training for comparison purposes.

**Transfer Learning :**

The final classification layer from the pretrained neural nets is replaced with a new fully connected layer that outputs the number of classes in the flower dataset.

The weights of the original pretrained model are frozen and training is only performed on the new fully connected layer. This way, the feature extraction section is kept untouched and only the classification layer is fine tuned. This is done to prevent the pre-trained weights which have learned used features from being overwritten and also to speed up the training process.

A suitable optimizer and loss function are chosen, and the hyperparameters are adjusted to obtain desirable accuracy levels.


## Evaluation Measure

Performance evaluation is based on Training accuracy, classification report containing accuracy_score, precision_score and recall_score, Confusion matrix and time taken for training of the model.

## Acknowledgements/References:

- https://www.kaggle.com/datasets/alxmamaev/flowers-recognition
- https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub
- https://www.analyticsvidhya.com/blog/2021/11/transfer-learning-with-tensorflow/
- Lecture Notes from CSI 5140
- https://www.wikipedia.org/






