import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import plot_roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np


def create_one_dataset_from_pairs(images_gabor_filters, pairs, is_true_pairs):
    pairs_distance = pd.DataFrame(columns=['features', 'label'])

    united_feature_list = []
    for i, row in pairs.iterrows():
        image1 = row['image1']
        image2 = row['image2']
        features_image1 = images_gabor_filters[images_gabor_filters['image'] == image1]['bytes'].to_numpy()
        features_image2 = images_gabor_filters[images_gabor_filters['image'] == image2]['bytes'].to_numpy()
        united_features = np.array(features_image1[0]) - np.array(features_image2[0])
        united_feature_list.append(united_features)

    pairs_distance['features'] = united_feature_list
    if is_true_pairs:
        pairs_distance['label'] = ["True" for i in range(len(pairs))]
    else:
        pairs_distance['label'] = ["False" for i in range(len(pairs))]

    return pairs_distance


def create_pairs_distances(images_gabor_filters, true_pairs_subset, impostor_pairs_subset):

    true_pairs_distances = create_one_dataset_from_pairs(images_gabor_filters, true_pairs_subset, True)
    impostor_pairs_distances = create_one_dataset_from_pairs(images_gabor_filters, impostor_pairs_subset, False)

    return true_pairs_distances, impostor_pairs_distances


def evaluate_with_svm(images_gabor_filters, true_pairs, impostor_pairs):
    true_pairs_subset = true_pairs.head(5000)
    impostor_pairs_subset = impostor_pairs.head(5000)

    true_pairs_distances, impostor_pairs_distances = \
        create_pairs_distances(images_gabor_filters, true_pairs_subset, impostor_pairs_subset)

    # ------------------------------SVM ---------------------------

    final_df = pd.concat([true_pairs_distances, impostor_pairs_distances], ignore_index=True)
    final_df['features'] = final_df['features'].apply(lambda x: np.array(x))

    X_train, X_test, y_train, y_test = train_test_split(final_df['features'], final_df['label'],
                                                        test_size=0.33, shuffle=True, random_state=101)

    clf = SVC()
    clf.fit(list(X_train), y_train)

    y_pred = clf.predict(list(X_test))
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    svc_disp = plot_roc_curve(clf, list(X_test), y_test)
    plt.grid(True)
    plt.show()

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
