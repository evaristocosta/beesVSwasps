
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.color import rgb2gray
from skimage.feature import hog
from PIL import Image
import numpy as np
import pandas as pd
import os


import matplotlib.pyplot as plt
from pathlib import Path


def get_image(row_id, root="datasets/"):
    """
        Opens the image, and returns the image as a numpy array.
    """
    filename = row_id.replace("\\", "/")
    file_path = os.path.join(root, filename)
    img = Image.open(file_path)
    return np.array(img)


def create_features(img):
    # resize
    img = img.resize([150, 150])
    # flatten three channel color image
    color_features = img.flatten()
    # convert image to grayscale
    gray_image = rgb2gray(img)
    # get HOG features from grayscale image
    hog_features = hog(gray_image, block_norm='L2-Hys',
                       pixels_per_cell=(16, 16))
    # combine color and hog features into a single array
    flat_features = np.hstack((color_features, hog_features))
    return flat_features


def create_feature_matrix(label_dataframe, total_dados):
    features_list = []
    # contadores de limite de dados
    total_bees = 0
    total_wasp = 0

    for _, row in label_dataframe.iterrows():
        # só considera casos onde é vespa ou abelha e com qualidade de foto boa
        if ((row.is_wasp == 1 and total_wasp < total_dados) or (row.is_bee == 1 and total_bees < total_dados)) and row.photo_quality == 1:
            if row.is_wasp == 1:
                total_wasp = total_wasp + 1
            else:
                total_bees = total_bees + 1

            # load image
            img = get_image(row.path)
            # get features for image
            image_features = create_features(img)
            features_list.append(image_features)

    # convert list of arrays into a matrix
    """ feature_matrix = np.zeros([len(features_list), len(
        max(features_list, key=lambda x: len(x)))])
    for i, j in enumerate(features_list):
        feature_matrix[i][0:len(j)] = j """

    return features_list


def load_bvsw(total_dados=50):
    print("Abrindo arquivo...")
    labels = pd.read_csv("datasets/labels.csv", index_col=0)

    # verifica limite de dados
    if total_dados > 4000:
        total_dados = 4000

    total_dados = total_dados // 2

    print("Criando matriz de features...")
    feature_matrix = create_feature_matrix(labels, total_dados)

    # define standard scaler
    ss = StandardScaler()
    # run this on our feature matrix
    print("Padronizando dados...")
    bees_stand = ss.fit_transform(feature_matrix)

    pca = PCA(n_components=min(500, total_dados*2))
    # use fit_transform to run PCA on our standardized matrix
    print("Rodando PCA (esse processo pode demorar um pouco)...")
    bees_pca = pca.fit_transform(bees_stand)
    # look at new shape
    print('Matriz de PCA concluída com formato: ', bees_pca.shape)

    classes = np.concatenate((np.ones(total_dados), np.zeros(total_dados)))

    return bees_pca, classes


if __name__ == "__main__":
    load_bvsw()
