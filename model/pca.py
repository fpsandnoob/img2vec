from sklearn.decomposition import PCA
from utils import *


def pca(embedding_dim, data):
    pca_model = PCA(embedding_dim)
    img = np.reshape(data, (len(data), -1))
    y = pca_model.fit_transform(img)
    return y
