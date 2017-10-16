from sklearn.decomposition import PCA
from utils import *


def pca(embedding_dim, data_path):
    img = []
    images = os.listdir(data_path)
    for i in images:
        if i is not "." or "..":
            img.append(image2matrix(os.path.join(data_path, i)))
    pca_model = PCA(embedding_dim)
    img = np.reshape(img, (len(images) - 2, -1))
    y = pca_model.fit_transform(img)
    return y
