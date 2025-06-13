import base64
from io import BytesIO
from PIL import Image, ImageEnhance
# import cv2
import numpy as np
import io
def convert_to_base64(pil_image):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """

    # buffered = BytesIO()
    # pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(pil_image).decode("utf-8")
    return img_str


# def enhance_light(image):

#     # Brightness
#     enhancer = ImageEnhance.Brightness(image)
#     bright_img = enhancer.enhance(3)  

#     # Contrast
#     enhancer = ImageEnhance.Contrast(bright_img)
#     contrast_img = enhancer.enhance(0.8)

#     return contrast_img


# def cek_brightness_histogram(image, threshold=72):
    
#     # image_np = np.frombuffer(image, np.uint8) 
#     # image = cv2.imdecode(image_np, cv2.IMREAD_COLOR) 
    
#     pil_cv = np.array(image)
#     pil_cv = cv2.cvtColor(pil_cv, cv2.COLOR_RGB2BGR)
    
#     hsv = cv2.cvtColor(pil_cv, cv2.COLOR_BGR2HSV)
#     v_channel = hsv[:, :, 2]
#     return np.mean(v_channel) < threshold

# def reduce_noise(image):
#     img = cv2.resize(image, (600, 600))
#     img = cv2.bilateralFilter(img, d=4, sigmaColor=70, sigmaSpace=70)
#     _, buffer = cv2.imencode('.jpg', img)
#     img_bytes = buffer.tobytes()
#     return img_bytes   

# def processor_image(img):
#     img = Image.open(io.BytesIO(img)) 
#     if cek_brightness_histogram(img):
#         img = enhance_light(img)
#         img = np.array(img)
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         img = reduce_noise(img)
#     else:    
#         img = np.array(img)
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         img = reduce_noise(img)
   
#     return img     
