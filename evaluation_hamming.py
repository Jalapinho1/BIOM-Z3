import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import plot_roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

def create_one_dataset_from_pairs(gabor_filter_banks, pairs, is_true_pairs):
    pairs_distance = pd.DataFrame(columns=['features', 'label'])

    united_feature_list = []
    for i, row in pairs.iterrows():
        image1 = row['image1']
        image2 = row['image2']
        features_image1 = gabor_filter_banks[gabor_filter_banks['image'] == image1]['bytes'].to_numpy()
        features_image2 = gabor_filter_banks[gabor_filter_banks['image'] == image2]['bytes'].to_numpy()
        united_features = np.array(features_image1[0]) - np.array(features_image2[0])
        united_feature_list.append(united_features)

    pairs_distance['features'] = united_feature_list
    if is_true_pairs:
        pairs_distance['label'] = ["True" for i in range(len(pairs))]
    else:
        pairs_distance['label'] = ["False" for i in range(len(pairs))]

    return pairs_distance


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

            hamming_distance = 0
            equal_pixels_num = 0
            not_equal_pixels_num = 0
            for p in range(thresholded_image1.shape[0]):
                for q in range(thresholded_image1.shape[1]):
                    if thresholded_image1[p][q] != thresholded_image2[p][q]:
                        if mask1_rotated[p][q] != 0 and mask2[p][q] != 0:
                            hamming_distance += 1
                            not_equal_pixels_num += 1
                    else:
                        if mask1_rotated[p][q] != 0 and mask2[p][q] != 0:
                            equal_pixels_num += 1

            pomer = equal_pixels_num / not_equal_pixels_num
            image_hamming_distances.append(hamming_distance)

        minimal_distance = min(image_hamming_distances)
        hamming_distances.append(minimal_distance)

    return hamming_distances


def evaluate_with_hamming(images, masks, true_pairs, impostor_pairs):
    true_pairs_subset = true_pairs.head(300)
    impostor_pairs_subset = impostor_pairs.head(300)

    sizes = [(61, 61), (51, 51), (41, 41), (31, 31), (21, 21)]

    for ksize in sizes:
        for theta in range(8):
            theta = theta / 8 * np.pi
            g_kernel = cv2.getGaborKernel(ksize, 6.0, theta, 8.0, 0.5, 0, ktype=cv2.CV_32F)
            print(f"G Kernel -> KSize={ksize}, Theta={theta}")

            true_pairs_distances = calculate_hamming_distance(images, masks, true_pairs_subset, g_kernel)
            impostor_pairs_distances = calculate_hamming_distance(images, masks, impostor_pairs_subset, g_kernel)

            plt.hist([true_pairs_distances, impostor_pairs_distances], color=['r', 'b'],
                     label=['True', 'Impostor'], bins=20, alpha=0.5)
            plt.show()
            # sns.distplot(true_pairs_distances, bins=50)
            # plt.title('True pair distance histogram')
            # plt.grid(True)
            # plt.show()
            # sns.distplot(impostor_pairs_distances, bins=50)
            # plt.title('Impostor pair distance histogram')
            # plt.grid(True)
            # plt.show()
            print('round done')



