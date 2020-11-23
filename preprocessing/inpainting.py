# Source: htps://zengxianyu.github.io/iic/#web-app
# Paper: High-Resolution Image Inpainting with Iterative Confidence Feedback and Guided Upsampling, ECCV 2020

import base64
from io import BytesIO
from PIL import Image
import requests
import pdb

img = Image.open("/home/socialab157/Desktop/YLD_fig/orig_11_20_0.jpg")
mask = Image.open("/home/socialab157/Desktop/YLD_fig/mask_11_20_0.jpg")

mode_img = img.mode
mode_msk = mask.mode

W, H = img.size
str_img = img.tobytes().decode("latin1")
str_msk = mask.tobytes().decode("latin1")

data = {'str_img': str_img, 'str_msk': str_msk, 'width':W, 'height':H,
        'mode_img':mode_img, 'mode_msk':mode_msk}

r = requests.post('http://47.57.135.203:2333/api', json=data)

str_result = r.json()['str_result']

result = str_result.encode("latin1")
result = Image.frombytes('RGB', (W, H), result, 'raw')
result.save("4_result.jpg")

