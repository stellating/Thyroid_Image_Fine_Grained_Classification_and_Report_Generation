"""Utilities
"""
import re
import base64

import numpy as np

from PIL import Image
from io import BytesIO
from scipy.misc import imread, imresize


def base64_to_pil(img_base64):
    """
    Convert base64 image data to PIL image
    """
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    # pil_image = Image.open(BytesIO(base64.b64decode(image_data)))
    read_image = imread(BytesIO(base64.b64decode(image_data)))
    return read_image


def np_to_base64(img_np):
    """
    Convert numpy image (RGB) to base64 string
    """
    img = Image.fromarray(img_np.astype('uint8'), 'RGB')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return u"data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("ascii")

