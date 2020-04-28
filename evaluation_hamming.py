import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import plot_roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from scipy.spatial import distance
from scipy.integrate import simps
from numpy import trapz
from sklearn import metrics


def calculate_hamming_distance(images, masks, pairs, g_kernel):
    hamming_distances = []
    for i, row in pairs.iterrows():
        image1 = images[images['image'] == row['image1']]['bytes'].to_numpy()[0]
        image2 = images[images['image'] == row['image2']]['bytes'].to_numpy()[0]

        mask1 = masks[masks['image'] == row['image1']]['bytes'].to_numpy()[0]
        mask2 = masks[masks['image'] == row['image2']]['bytes'].to_numpy()[0]

        # plt.imshow(image1_array)
        # plt.show()

        rolling_param = [-30, 0, 30]
        image_hamming_distances = []
        for rotation in rolling_param:
            image1_rotated = np.roll(image1, rotation).copy()
            mask1_rotated = np.roll(mask1, rotation).copy()

            filtered_image1 = cv2.filter2D(image1_rotated, cv2.CV_8UC3, g_kernel)
            filtered_image2 = cv2.filter2D(image2, cv2.CV_8UC3, g_kernel)

            ret1, thresholded_image1 = cv2.threshold(filtered_image1, 127, 255, cv2.THRESH_BINARY)
            ret2, thresholded_image2 = cv2.threshold(filtered_image2, 127, 255, cv2.THRESH_BINARY)

            # Visualize filter

            # plt.imshow(image1_rotated)
            # plt.show()
            # plt.imshow(thresholded_image1)
            # plt.show()

            # Option 1

            thresholded_image1[mask1_rotated == 0] = [0]
            thresholded_image2[mask2 == 0] = [0]

            dist = distance.hamming(thresholded_image1.ravel(), thresholded_image2.ravel())
            image_hamming_distances.append(dist)

            # Option 2

            # hamming_distance = 0
            # equal_pixels_num = 0
            # not_equal_pixels_num = 0
            # for p in range(thresholded_image1.shape[0]):
            #     for q in range(thresholded_image1.shape[1]):
            #         if thresholded_image1[p][q] != thresholded_image2[p][q]:
            #             if mask1_rotated[p][q] != 0 and mask2[p][q] != 0:
            #                 hamming_distance += 1
            #                 not_equal_pixels_num += 1
            #         else:
            #             if mask1_rotated[p][q] != 0 and mask2[p][q] != 0:
            #                 equal_pixels_num += 1
            #
            # pomer = equal_pixels_num / not_equal_pixels_num
            # image_hamming_distances.append(hamming_distance)

        minimal_distance = min(image_hamming_distances)
        hamming_distances.append(minimal_distance)

    return hamming_distances


def make_roc(true_pairs_distances, impostor_pairs_distances):
    tpr_list = []
    fpr_list = []
    minimum = min([min(true_pairs_distances), min(impostor_pairs_distances)])
    maximum = max([max(true_pairs_distances), max(impostor_pairs_distances)])
    threshold = 0
    for threshold in np.linspace(minimum, maximum, 200):
        tpr = len([i for i in true_pairs_distances if i < threshold]) / len(true_pairs_distances)
        fpr = len([i for i in impostor_pairs_distances if i < threshold]) / len(impostor_pairs_distances)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    auc = metrics.auc(fpr_list, tpr_list)
    print(f"AUC={auc}")

    plt.figure()
    lw = 2
    plt.plot(fpr_list, tpr_list, color='darkorange',
             lw=lw, label=f"AUC = {round(auc, 4)}")
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic 1')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def calculate_hamming_and_roc(ksize, sigma, theta, lambda_i, gamma, images, masks, true_pairs, impostor_pairs):

    g_kernel = cv2.getGaborKernel(ksize, sigma, theta, lambda_i, gamma, 0, ktype=cv2.CV_32F)
    print(f"G Kernel -> KSize={ksize}, Sigma={sigma}, Theta={theta}, Lambda={lambda_i}, Gamma={gamma}")


    true_pairs_distances = calculate_hamming_distance(images, masks, true_pairs, g_kernel)
    impostor_pairs_distances = calculate_hamming_distance(images, masks, impostor_pairs, g_kernel)

    plt.hist([true_pairs_distances, impostor_pairs_distances], color=['r', 'b'],
             label=['True', 'Impostor'], bins=20, alpha=0.5)
    plt.legend()
    plt.title(f"G Kernel -> S={sigma}, T={theta}, "f"L={lambda_i}, G={gamma}")
    plt.show()

    make_roc(true_pairs_distances, impostor_pairs_distances)


def try_different_g_kernels(images, masks, true_pairs_subset, impostor_pairs_subset):
    sizes = [(51, 51)]
    sigmas = [4.0, 5.0, 6.0]
    lambdas = [10.0, 11.0, 12.0]
    gammas = [0.4, 0.5, 0.6]

    for ksize in sizes:
        for sigma in sigmas:
            for theta in range(1):
                theta = 2 / 4 * np.pi
                for lambda_i in lambdas:
                    for gamma in gammas:
                        g_kernel = cv2.getGaborKernel(ksize, sigma, theta, lambda_i, gamma, 0, ktype=cv2.CV_32F)
                        print(f"G Kernel -> KSize={ksize}, Sigma={sigma}, Theta={theta}, "
                              f"Lambda={lambda_i}, Gamma={gamma}")

                        true_pairs_distances = calculate_hamming_distance(images, masks, true_pairs_subset, g_kernel)
                        impostor_pairs_distances = calculate_hamming_distance(images, masks, impostor_pairs_subset,
                                                                              g_kernel)

                        plt.hist([true_pairs_distances, impostor_pairs_distances], color=['r', 'b'],
                                 label=['True', 'Impostor'], bins=20, alpha=0.5)
                        plt.legend()
                        plt.title(f"G Kernel -> S={sigma}, T={theta}, "
                                  f"L={lambda_i}, G={gamma}")
                        plt.show()


def evaluate_with_hamming(images, masks, true_pairs, impostor_pairs):
    true_pairs_subset = true_pairs.head(1000)
    impostor_pairs_subset = impostor_pairs.head(1000)

    # try_different_g_kernels(images, masks, true_pairs_subset, impostor_pairs_subset)

    ksize = (51, 51)
    sigma = 6.0
    theta = 1/2 * np.pi
    lambda_i = 12.0
    gamma = 0.6
    calculate_hamming_and_roc(ksize, sigma, theta, lambda_i, gamma, images, masks,
                               true_pairs_subset, impostor_pairs_subset)




