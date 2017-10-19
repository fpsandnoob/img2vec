from utils import *
from model.autoencoder import autoencoder
from model.pca import pca
from traceback import print_exc
import os


class Img2Vec:
    def __init__(self):
        self.embedding_dim = 256
        self.method = "pca"
        self.image_dim = (36, 36)
        self._method = ["pca", "autoencoder"]
        self.font_path = "./fonts/NotoSansCJKsc-Regular.otf"
        self._data = None
        self._loaded = False
        self.pretrained_embedding = r"pretrained/pca.npy"

    def build_char_dataset(self, char_dict, image_dim=None, language=None, font_path=None):
        if image_dim is not None:
            if not isinstance(image_dim, tuple):
                raise TypeError("The dimension of images must be tuple not {}!".format(type(image_dim)))
            if len(image_dim) != 2:
                raise ValueError("The tuple must be 2-D tuple not {}-D tuple!".format(len(image_dim)))
            self.image_dim = image_dim
        if font_path is not None:
            if not isinstance(font_path, str):
                raise TypeError("The path of font must be string not {}!".format(type(font_path)))
            if not os.path.exists(font_path):
                raise OSError("{} does not exist!".format(font_path))
            else:
                self.font_path = font_path
        print(font_path)
        return build_char_data(char_dict, image_dim, self.font_path)

    def _train(self, data):
        result = []
        if self.method == "pca":
            result = pca(self.embedding_dim, data)
        elif self.method == "autoencoder":
            result = autoencoder(self.embedding_dim, data)
        assert result is not None
        return result

    def fit(self, data, embedding_dim=None, method=None):
        if embedding_dim is not None:
            self.embedding_dim = embedding_dim
        if method is not None:
            self.method = embedding_dim

        if not isinstance(self.method, str):
            raise TypeError("Method must be string not {}!".format(type(self.method)))
        if not isinstance(self.embedding_dim, int):
            raise TypeError("Embedding dimension must be int not {}!".format(type(self.embedding_dim)))
        if self.method not in self._method:
            raise ValueError("{} is not supported!".format(self.method))
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("The matrices of images must be list not {}!".format(type(data)))
        else:
            raise ValueError("Data should not be Null!")
        self._data = self._train(data)
        self._loaded = True

    def fit_transform(self, data, embedding_dim=None, method=None):
        self.fit(data, embedding_dim, method)
        if self._loaded:
            return self._data

    @staticmethod
    def _read_dir(path):
        data = []
        files = os.listdir(path)
        for f in files:
            data.append(image2matrix(f))
        return data

    def read_img_from_dir(self, path):
        self._check_path(path)
        return self._read_dir(path)

    @staticmethod
    def _check_path(path):
        if path is not None:
            if not isinstance(path, str):
                raise TypeError("The path must be string not {}!".format(type(path)))
        else:
            raise ValueError("Path should not be Null!")

    def load(self, path):
        try:
            self._check_path(path)
        except ValueError:
            path = self.pretrained_embedding
        np.load(path, self._data)
        self._loaded = True

    def save(self, path):
        self._check_path(path)
        if self._loaded:
            np.save(path, self._data)
        else:
            raise ValueError("Data is Null, can't be exported!")
