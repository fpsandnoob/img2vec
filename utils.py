import traceback
import os
import uuid
import numpy as np
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


def build_char_data(char_dict, image_dim, font_path):
    _check_x(char_dict)
    font = ImageFont.truetype(font_path, 29)
    if not os.path.exists("./data"):
        os.mkdir("./data")
    path = os.path.join("./data", uuid.uuid1().hex)
    try:
        os.mkdir(path)
    except OSError:
        traceback.print_exc()
        path = os.path.join("./data", uuid.uuid1().hex)
        os.mkdir(path)
    chars = char_dict.keys()
    for char in chars:
        base = Image.new('RGBA', image_dim, (255, 255, 255, 0))
        txt = Image.new('RGBA', image_dim, (255, 255, 255, 0))
        d = ImageDraw.Draw(txt)
        d.text((0, 0), char, font=font, fill=(0, 0, 0, 255))
        out = Image.alpha_composite(base, txt)
        out = out.convert("L")
        out.save(os.path.join(path, "{}.png".format(char)))
    return path


def image2matrix(filename):
    im = Image.open(filename)
    width, height = im.size
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data, dtype='float') / 255.0
    new_data = np.reshape(data, (height, width))
    return new_data
