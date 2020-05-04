from math import floor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from skimage.feature import hog
from skimage.exposure import exposure
import cv2
from cv2 import imread
from keras.models import Model


def extract_feautures_cnn_vgg(images, masks):
    encoded_images = pd.DataFrame(columns=['feature_vector'])

    model = VGG16(weights='imagenet', include_top=False)
    model.summary()

    model2 = Model(model.input, model.layers[-16].output)
    model2.summary()

    layer_names = []
    for layer in model.layers[:12]:
        layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot

    vgg16_feature_list = []

    for i, row in images.iterrows():
        img_data = row['bytes']
        mask_data = masks.iloc[i]['bytes']
        img_data[mask_data == 0] = [0]

        # 3 channels greyscale
        multiple_channel = np.stack((img_data,) * 3, axis=-1)

        # plt.imshow(multiple_channel)
        # plt.show()

        img_data = image.img_to_array(multiple_channel)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        vgg16_feature = model.predict(img_data)
        vgg16_feature_np = np.array(vgg16_feature)
        vgg16_feature_list.append(vgg16_feature_np.flatten())
        encoded_images.loc[row['image'], 'feature_vector'] = vgg16_feature_np.flatten()

    return encoded_images


def extract_feautures_cnn_resnet(images, masks):
    encoded_images = pd.DataFrame(columns=['feature_vector'])

    model = ResNet50(weights='imagenet', pooling=max, include_top=False)
    model.summary()

    model2 = Model(model.input, model.layers[-163].output)
    model2.summary()

    layer_names = []
    for layer in model.layers[:12]:
        layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot

    for i, row in images.iterrows():
        img_data = row['bytes']
        mask_data = masks.iloc[i]['bytes']
        img_data[mask_data == 0] = [0]

        # 3 channels greyscale
        multiple_channel = np.stack((img_data,) * 3, axis=-1)

        # plt.imshow(multiple_channel)
        # plt.show()

        img_data = image.img_to_array(multiple_channel)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        resnet_feature = model2.predict(img_data)
        resnet_feature_np = np.array(resnet_feature)
        encoded_images.loc[row['image'], 'feature_vector'] = resnet_feature_np.flatten()

    return encoded_images