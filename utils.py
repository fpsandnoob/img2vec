import traceback
import os
import uuid
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    traceback.print_exc()
    print(ImportError("For python3 use \"pip install pillow\".\n"
                      "For python2 use \"pip install pil\". "))


def _check_x(x):
    if x.keys() is None:
        raise ValueError("The key of X is Null!")
    if x.items() is None:
        raise ValueError("The item of X is Null!")


def build_char_data(char_dict, image_dim, language, font_path):
    _check_x(char_dict)
    font = ImageFont.truetype(font_path, 36)
    if not os.path.exists("./data"):
        os.mkdir("./data")
    path = os.path.join("./data", uuid.uuid1())
    try:
        os.mkdir(path)
    except OSError:
        traceback.print_exc()
        path = os.path.join("./data", uuid.uuid1())
        os.mkdir(path)
    chars = char_dict.keys()
    for char in chars:
        txt = Image.new('RGBA', (36, 36), )

