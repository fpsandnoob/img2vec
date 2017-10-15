from utils import *
import os


class Img2Vec:
    def __init__(self):
        self.language = "sc"
        self.embedding_dim = 256
        self.method = "pca"
        self.image_dim = (36, 36)
        self._method = ["pca", "autoencoder"]
        self.font_path = "./fonts/NotoSansCJK{}-Regular.otf".format(self.language)

    def build_char_dataset(self, char_dict, image_dim=None, language=None, font_path=None):
        if image_dim is not None:
            if not isinstance(image_dim, tuple):
                raise TypeError("The dimension of images must be tuple not {}!".format(type(image_dim)))
            if len(image_dim) != 2:
                raise ValueError("The tuple must be 2-D tuple not {}-D tuple!".format(len(image_dim)))
            self.image_dim = image_dim
        if language is not None:
            if not isinstance(language, str):
                raise TypeError("The language must be string not {}!".format(type(language)))
            else:
                self.language = language
        if font_path is not None:
            if not isinstance(font_path, str):
                raise TypeError("The path of font must be string not {}!".format(type(font_path)))
            if not os.path.exists(font_path):
                raise OSError("{} does not exist!".format(font_path))
            else:
                self.font_path = font_path


    def _train(self):
        if self.method == "pca":
            pass

    def fit(self, embedding_dim=None, method=None):
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

        self._train()

    def load(self):
        pass

    def save(self):
        pass

    def wv(self):
        pass
