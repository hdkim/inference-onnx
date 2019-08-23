from PIL import Image
import io


def image_to_byte_array(path):
  image = Image.open(path)
  imgByteArr = io.BytesIO()
  image.save(imgByteArr, format=image.format)
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr