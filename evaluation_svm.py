import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import plot_roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np


def create_one_dataset_from_pairs(features, pairs, feature_type, is_true_pairs):
    pairs_distance = pd.DataFrame(columns=['features', 'label'])

    united_feature_list = []
    for i, row in pairs.iterrows():
        image1 = row['image1']
        image2 = row['image2']
        if feature_type == "GaborFilters":
            features_image1 = features[features['image'] == image1]['bytes'].to_numpy()
            features_image2 = features[features['image'] == image2]['bytes'].to_numpy()
            united_features = np.array(features_image1[0]) - np.array(features_image2[0])
        else:
            features_image1 = features.loc[image1]['feature_vector']
            features_image2 = features.loc[image2]['feature_vector']
            united_features = features_image1 - features_image2
        # united_features = np.concatenate((np.array(features_image1), np.array(features_image2)), axis=0)
        united_feature_list.append(united_features)

    pairs_distance['features'] = united_feature_list
    if is_true_pairs:
        pairs_distance['label'] = ["True" for i in range(len(pairs))]
    else:
        pairs_distance['label'] = ["False" for i in range(len(pairs))]

    return pairs_distance


def create_pairs_distances(images_gabor_filters, true_pairs_subset, impostor_pairs_subset, feature_type):

    true_pairs_distances = create_one_dataset_from_pairs(images_gabor_filters, true_pairs_subset, feature_type, True)
    impostor_pairs_distances = create_one_dataset_from_pairs(images_gabor_filters, impostor_pairs_subset, feature_type, False)

    return true_pairs_distances, impostor_pairs_distances


def evaluate_with_svm(images_features, true_pairs, impostor_pairs, feature_type):
    true_pairs_subset = true_pairs.head(4000)
    impostor_pairs_subset = impostor_pairs.head(4000)

    true_pairs_distances, impostor_pairs_distances = \
        create_pairs_distances(images_features, true_pairs_subset, impostor_pairs_subset, feature_type)

    # ------------------------------SVM ---------------------------

    final_df = pd.concat([true_pairs_distances, impostor_pairs_distances], ignore_index=True)
    final_df['features'] = final_df['features'].apply(lambda x: np.array(x))

    X_train, X_test, y_train, y_test = train_test_split(final_df['features'], final_df['label'],
                                                        test_size=0.33, shuffle=True, random_state=101)

    clf = SVC()
    clf.fit(list(X_train), y_train)

    y_pred = clf.predict(list(X_test))

    # if feature_type == "GaborFilters":
    #     clf.fit(list(X_train), y_train)
    # else:
    #     clf.fit(np.array(X_train).reshape(-1, 1), y_train)

    # if feature_type == "GaborFilters":
    #     y_pred = clf.predict(list(X_test))
    # else:
    #     y_pred = clf.predict(np.array(X_test).reshape(-1, 1))

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    # if feature_type == "GaborFilters":
    #     svc_disp = plot_roc_curve(clf, list(X_test), y_test)
    # else:
    #     svc_disp = plot_roc_curve(clf, np.array(X_test).reshape(-1, 1), y_test)
    svc_disp = plot_roc_curve(clf, list(X_test), y_test)
    plt.grid(True)
    plt.show()

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


