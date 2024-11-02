# Plant-Disease-Detection
## 1. Project Description

The goal of this project is to develop a deep learning model that can accurately classify Plant Diseases and predict the true class. We will use simple convolutional neural networks (CNNs), EfficientNet, ResNet, and VGG19 to train our models, as they have been shown to be effective in image classification tasks. the data contain 38 classes of plant disease.

![image](https://github.com/user-attachments/assets/6575be71-82f6-4416-98b4-3f1507085b8d)

## 2. Table of Contents
[Dataset](https://github.com/elnemr19/Plant-Disease-Detection/tree/main?tab=readme-ov-file#3-dataset)

[Preprocessing](https://github.com/elnemr19/Plant-Disease-Detection/tree/main?tab=readme-ov-file#4-preprocessing)

[Model Overview](https://github.com/elnemr19/Plant-Disease-Detection/tree/main?tab=readme-ov-file#5model-overview)

[Results](https://github.com/elnemr19/Plant-Disease-Detection/tree/main?tab=readme-ov-file#6-results)

[Deployment](https://github.com/elnemr19/Plant-Disease-Detection/tree/main?tab=readme-ov-file#7-deployment)


## 3. Dataset

**Source** : Kaggle - [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
**Description** :the data contain the folders, Train - Validation, which is categorized into 38 different classes

## 4. Preprocessing
In this process i resize my images into 100 X 100 ,and i make augmentation on my data by do rotation by 50, zoom by 0.2, rescale by 1/255.0, and split my training data into train and validation by 
90 % - 10 % 

I also use flow_from_directory to read and processing my data by batches, beacause my data is large and i can't add it into listes



## 5.Model Overview


## 6. Results

![kk](https://github.com/user-attachments/assets/c734fea6-645b-426b-a58c-ba805632ce7a)



## 7. Deployment

I use streamlit to make user interface 

link: [Plant Disease App](http://192.168.1.51:8501)


![image](https://github.com/user-attachments/assets/cafa45d2-c01d-4952-96a0-07f53d7b59f1)


