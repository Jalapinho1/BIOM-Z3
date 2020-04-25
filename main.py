import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cv2 import imread, imshow, waitKey, CascadeClassifier, rectangle
import cv2
from PIL import Image
from math import floor, sqrt
from scipy.stats import kurtosis, skew


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
            Xc = (1 - r_pro) * Xi + r_pro * Xo
            Yc = (1 - r_pro) * Yi + r_pro * Yo

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
            color = image[int(Yc)][int(Xc)]  # color of the pixel

            mask[j][i] = mask_color
            flat[j][i] = color
    return flat, mask


def create_gabor_filter_bank(image):
    bank = pd.DataFrame()
    bank['Original image'] = image.reshape(-1)
    num = 1
    for theta in range(4):
        theta = theta / 4 * np.pi
        for sigma in (3, 5):
            for lambd in np.arange(np.pi / 4, np.pi, np.pi / 4):
                for gamma in (0.25, 0.5):
                    label = 'Gabor' + str(num)

                    # Apply Gabor Kernel on the image
                    g_kernel = cv2.getGaborKernel((21, 21), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
                    filtered_image = cv2.filter2D(image, cv2.CV_8UC3, g_kernel)
                    plt.imshow(g_kernel)
                    plt.show()
                    plt.imshow(filtered_image)
                    plt.show()

                    # Thresholding the image
                    ret, thresholded_image = cv2.threshold(filtered_image, 127, 255, cv2.THRESH_BINARY)
                    # plt.imshow(thresholded_image)
                    # plt.show()

                    # Add filter to the bank
                    reshaped_filtered_img = thresholded_image.reshape(-1)
                    bank[label] = reshaped_filtered_img

                    num += 1
    return bank


def main():
    annotation = pd.read_csv('duhovky/iris_annotation.csv')
    print(annotation)
    for i, row in annotation.iterrows():
        file = 'duhovky/' + str(row['image'])
        img = imread(file)

        # eye_center_1 = tuple([row[' center_x_1'], row[' center_y_1']])
        # cv2.circle(img, eye_center_1, row[' polomer_1'], (255,0,0), thickness=2)
        # eye_center_2 = tuple([row[' center_x_2'], row[' center_y_2']])
        # cv2.circle(img, eye_center_2, row[' polomer_2'], (255, 0, 0), thickness=2)

        image, mask = daugman_normalizaiton(img, row, 60, 360)

        # plt.imshow(image, cmap='gray')
        # plt.show()
        # plt.imshow(mask, cmap='gray')
        # plt.show()

        # non_black_pixels_mask = np.any(mask != [0, 0, 0], axis=-1)
        # black_pixels_mask = np.all(mask == [0, 0, 0], axis=-1)
        # image[black_pixels_mask] = [255,255,255]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.equalizeHist(image)
        # plt.imshow(image, cmap='gray')
        # plt.show()

        gabor_filters_bank = create_gabor_filter_bank(image)

        print('end of file')
        # g_kernel = cv2.getGaborKernel((151, 151), 6.0, theta, 10.0, 0.75, 0, ktype=cv2.CV_32F)
        # filtered_img = cv2.filter2D(image, cv2.CV_8UC3, g_kernel)

        # imshow('face detection', filtered_img)
        # waitKey()

        # h, w = g_kernel.shape[:2]
        # g_kernel = cv2.resize(g_kernel, (3 * w, 3 * h), interpolation=cv2.INTER_CUBIC)


main()