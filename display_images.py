from PIL import Image


def display_image(img_array):
    display = Image.fromarray(img_array)
    display.show()
