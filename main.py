import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cv2 import imread, imshow, waitKey, CascadeClassifier, rectangle
import cv2
from PIL import Image
from math import floor, sqrt
from scipy.stats import kurtosis, skew
import os
import shutil

def is_point_inside(dx, dy, R):
    if sqrt(dx ** 2 + dy ** 2) < R:
        return True
    else:
        return False


def daugman_normalizaiton(image, row, height, width):
    center_x_1 = row[' center_x_1']
    center_y_1 = row[' center_y_1']
    r_in = row[' polomer_1']
    center_x_2 = row[' center_x_2']
    center_y_2 = row[' center_y_2']
    r_out = row[' polomer_2']

    bottom_lid_x_1 = row[' center_x_3']
    bottom_lid_y_1 = row[' center_y_3']
    bottom_lid_radius = row[' polomer_3']
    upper_lid_x_2 = row[' center_x_4']
    upper_lid_y_2 = row['center_y_4']
    upper_lid_radius = row[' polomer_4']

    flat = np.zeros((height,width, 3), np.uint8)
    mask = np.zeros((height,width, 3), np.uint8)

    thetas = np.arange(0, 2 * np.pi, 2 * np.pi / width)  # Theta values
    for i in range(width):
        for j in range(height):
            theta = thetas[i]  # value of theta coordinate
            r_pro = j / height  # value of r coordinate(normalized)

            # get coordinate of boundaries
            Xi = center_x_1 + r_in * np.cos(theta)
            Yi = center_y_1 + r_in * np.sin(theta)
            Xo = center_x_2 + r_out * np.cos(theta)
            Yo = center_y_2 + r_out * np.sin(theta)

            # the matched cartesian coordinates for the polar coordinates
            Xc = floor((1 - r_pro) * Xi + r_pro * Xo)
            Yc = floor((1 - r_pro) * Yi + r_pro * Yo)

            upper_dx = abs(Xc - upper_lid_x_2)
            upper_dy = abs(Yc - upper_lid_y_2)
            bottom_dx = abs(Xc - bottom_lid_x_1)
            bottom_dy = abs(Yc - bottom_lid_y_1)
            upper_R = upper_lid_radius
            bottom_R = bottom_lid_radius

            inside_bottom_radius = is_point_inside(bottom_dx, bottom_dy, bottom_R)
            inside_upper_radius = is_point_inside(upper_dx, upper_dy, upper_R)

            if inside_bottom_radius and inside_upper_radius:
                mask_color = np.array([255, 255,255])
            else:
                mask_color = np.zeros(3)

            if Yc >= 280.0:
                color = image[279][int(Xc)]
            else:
                color = image[int(Yc)][int(Xc)]  # color of the pixel

            mask[j][i] = mask_color
            flat[j][i] = color
    return flat, mask


def create_gabor_filter_bank(image):
    bank = pd.DataFrame()
    bank['Original image'] = image.reshape(-1)
    num = 1
    sizes = [(61,61), (51,51), (41,41), (31,31), (21,21)]

    # local_energy = np.zeros((60,360))
    # mean_amplitude = np.zeros((60,360))
    local_energy = []
    mean_amplitude = []

    for ksize in sizes:
        for theta in range(8):
            theta = theta / 8 * np.pi

            label = 'Gabor' + str(num)
            # Apply Gabor Kernel on the image
            g_kernel = cv2.getGaborKernel(ksize, 6.0, theta, 8.0, 0.5, 0, ktype=cv2.CV_32F)
            filtered_image = cv2.filter2D(image, cv2.CV_8UC3, g_kernel)
            # plt.imshow(g_kernel)
            # plt.show()
            # plt.imshow(filtered_image)
            # plt.show()

            # Thresholding the image
            ret, thresholded_image = cv2.threshold(filtered_image, 127, 255, cv2.THRESH_BINARY)
            # plt.imshow(thresholded_image)
            # plt.show()

            # Add filter to the bank
            reshaped_filtered_img = thresholded_image.reshape(-1)
            bank[label] = reshaped_filtered_img

            # local_energy += np.square(thresholded_image.astype(float))
            local_energy.append(sum(sum(np.square(thresholded_image.astype(float)))))
            # mean_amplitude += np.abs(thresholded_image)
            mean_amplitude.append(sum(sum(np.abs(thresholded_image.astype(float)))))
            num += 1

    image_features = local_energy + mean_amplitude
    return bank, image_features


def read_images():
    annotation = pd.read_csv('duhovky/iris_annotation.csv')
    print(annotation)

    images_dict = {}
    masks_dict = {}

    for i, row in annotation.iterrows():
        file = 'duhovky/' + str(row['image'])
        img = imread(file)

        if not img is None:
            image, mask = daugman_normalizaiton(img, row, 60, 360)
            images_dict[row['image']] = image
            masks_dict[row['image']] = mask

            # plt.imshow(image, cmap='gray')
            # plt.show()
            # plt.imshow(mask, cmap='gray')
            # plt.show()

            # non_black_pixels_mask = np.any(mask != [0, 0, 0], axis=-1)
            # black_pixels_mask = np.all(mask == [0, 0, 0], axis=-1)
            # image[black_pixels_mask] = [255,255,255]

            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # image = cv2.equalizeHist(image)
            # plt.imshow(image, cmap='gray')
            # plt.show()

            # gabor_filters_bank, image_features = create_gabor_filter_bank(image)

    images = pd.DataFrame(images_dict.items(), columns=['image', 'bytes'])
    masks = pd.DataFrame(masks_dict.items(), columns=['image', 'bytes'])

    images.to_pickle('cartesian_images')
    masks.to_pickle('cartesian_masks')


def find_next_subject(df, index, image_number):
    while index < len(df) and df.iloc[index]['image'].strip().split("/")[0] == image_number:
        index += 1
    return index


def create_impostor_pairs(images):
    false_pairs = {}

    index = 0
    j_index = 0
    counter = 0
    while index < len(images) - 1:
        file_path = images.iloc[index]['image']
        subject = file_path.strip().split("/")[0]

        j_index = index
        while j_index < len(images):
            next_file_path = images.iloc[j_index]['image']
            next_subject = next_file_path.strip().split("/")[0]

            j_index = find_next_subject(images, j_index, next_subject)
            if j_index > len(images) - 2:
                break
            else:
                next_file_path = images.iloc[j_index]['image']
                next_subject = next_file_path.strip().split("/")[0]

            if subject != next_subject:
                image_pair = {
                    "image1": images.iloc[index]['image'],
                    "image2": images.iloc[j_index]['image'],
                }
                false_pairs[counter] = image_pair

                counter += 1
                j_index += 1
        index = find_next_subject(images, index, subject)

    false_pairs_df = pd.DataFrame.from_dict(false_pairs, orient='index')
    return false_pairs_df

def create_true_pairs(images):
    true_pairs = {}

    index = 0
    counter = 0
    while index < len(images) - 1:
        next_index = index + 1

        file_path = images.iloc[index]['image']
        subject = file_path.strip().split("/")[0]
        eye_side = file_path.strip().split("/")[1]

        next_file_path = images.iloc[next_index]['image']
        next_subject = next_file_path.strip().split("/")[0]
        next_eye_side = next_file_path.strip().split("/")[1]

        while subject == next_subject and eye_side == next_eye_side:
            image_pair = {
                "image1": images.iloc[index]['image'],
                "image2": images.iloc[next_index]['image'],
            }
            true_pairs[counter] = image_pair

            counter += 1
            next_index += 1
            if next_index > len(images) - 1:
                break
            next_subject = images.iloc[next_index]['image'].split("/")[0]
            next_eye_side = images.iloc[next_index]['image'].split("/")[1]

        if next_index > len(images) - 1:
            break

        index += 1

    true_pairs_df = pd.DataFrame.from_dict(true_pairs, orient='index')
    return true_pairs_df


def main():
    # read_images()

    images = pd.read_pickle('cartesian_images.csv')
    # print(images.head())
    # for i, row in images.iterrows():
    #     plt.imshow(row['bytes'])
    #     plt.show()

    # masks = pd.read_pickle('cartesian_masks.csv')
    # print(masks.head())
    # for i, row in masks.iterrows():
    #     plt.imshow(row['bytes'])
    #     plt.show()

    # true_pairs_df = create_true_pairs(images)
    # true_pairs_df.to_csv('true_pairs.csv',index=False)
    # impostor_pairs_df = create_impostor_pairs(images)
    # impostor_pairs_df.to_csv('impostor_pairs.csv',index=False)

    true_pairs_df = pd.read_csv('true_pairs.csv')
    impostor_pairs_df = pd.read_csv('impostor_pairs.csv')
    print(len(true_pairs_df))

main()