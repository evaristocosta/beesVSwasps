# used to change filepaths
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.color import rgb2gray
from skimage.feature import hog
from PIL import Image
import numpy as np
import pandas as pd
import os

import matplotlib as mpl
import matplotlib.pyplot as plt


# load the labels using pandas
labels = pd.read_csv("datasets/labels.csv", index_col=0)

# show the first five rows of the dataframe using head
print(labels.head())


def get_image(row_id, root="datasets/"):
    """
    Converts an image number into the file path where the image is located, 
    opens the image, and returns the image as a numpy array.
    """
    filename = row_id.replace("\\", "/")
    file_path = os.path.join(root, filename)
    img = Image.open(file_path)
    return np.array(img)


# subset the dataframe to just Apis (genus is 0.0) get the value of the sixth item in the index
apis_row = labels[labels.is_bee == 1.0].path[1]

# show the corresponding image of an Apis
plt.imshow(get_image(apis_row))
plt.show()

# subset the dataframe to just wasp (genus is 1.0) get the value of the sixth item in the index
wasp_row = labels[labels.is_wasp == 1.0].path[8122]

# show the corresponding image of a wasp
plt.imshow(get_image(wasp_row))
plt.show()


# load a wasp image using our get_image function and wasp_row from the previous cell
wasp = get_image(wasp_row)

# print the shape of the wasp image
print('Color wasp image has shape: ', wasp.shape)

# convert the wasp image to grayscale
gray_wasp = rgb2gray(wasp)

# show the grayscale image
#plt.imshow(gray_wasp, cmap=mpl.cm.gray)
#plt.show()

# grayscale wasp image only has one channel
print('Grayscale wasp image has shape: ', gray_wasp.shape)


# run HOG using our grayscale wasp image
hog_features, hog_image = hog(gray_wasp,
                              visualize=True,
                              block_norm='L2-Hys',
                              pixels_per_cell=(16, 16))

# show our hog_image with a gray colormap
#plt.imshow(hog_image, cmap=mpl.cm.gray)
#plt.show()


def create_features(img):
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


wasp_features = create_features(wasp)

# print shape of wasp_features
print(wasp_features.shape)


def create_feature_matrix(label_dataframe):
    features_list = []

    total_bees = 0
    total_wasp = 0
    for _, row in label_dataframe.iterrows():
        if ((row.is_wasp == 1 and total_wasp < 250) or (row.is_bee == 1 and total_bees < 250)) and row.photo_quality == 1:
            if row.is_wasp == 1:
                total_wasp = total_wasp + 1
            else:
                total_bees = total_bees + 1

            print(total_bees, total_wasp)
            print(row.path)
            # load image
            img = get_image(row.path)
            # get features for image
            image_features = create_features(img)
            features_list.append(image_features)

    # convert list of arrays into a matrix
    feature_matrix = np.zeros([len(features_list),len(max(features_list,key = lambda x: len(x)))])
    for i,j in enumerate(features_list):
        feature_matrix[i][0:len(j)] = j
    
    return feature_matrix


# run create_feature_matrix on our dataframe of images
feature_matrix = create_feature_matrix(labels)


# get shape of feature matrix
print('Feature matrix shape is: ', feature_matrix.shape)

# define standard scaler
ss = StandardScaler()
# run this on our feature matrix
bees_stand = ss.fit_transform(feature_matrix)

pca = PCA(n_components=500)
# use fit_transform to run PCA on our standardized matrix
bees_pca = pca.fit_transform(bees_stand)
# look at new shape
print('PCA matrix shape is: ', bees_pca.shape)

y = np.concatenate((np.ones(250), np.zeros(250)))

X_train, X_test, y_train, y_test = train_test_split(bees_pca,
                                                    y,
                                                    test_size=.3,
                                                    random_state=1234123)

# look at the distrubution of labels in the train set
print(pd.Series(y_train).value_counts())


# define support vector classifier
svm = SVC(kernel='linear', probability=True, random_state=42)

# fit model
svm.fit(X_train, y_train)


# generate predictions
y_pred = svm.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_pred, y_test)
print('Model accuracy is: ', accuracy)


# predict probabilities for X_test using predict_proba
probabilities = svm.predict_proba(X_test)

# select the probabilities for label 1.0
y_proba = probabilities[:,1]

# calculate false positive rate and true positive rate at different thresholds
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_proba, pos_label=1)

# calculate AUC
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')
# plot the false positive rate on the x axis and the true positive rate on the y axis
roc_plot = plt.plot(false_positive_rate,
                    true_positive_rate,
                    label='AUC = {:0.2f}'.format(roc_auc))

plt.legend(loc=0)
plt.plot([0,1], [0,1], ls='--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
