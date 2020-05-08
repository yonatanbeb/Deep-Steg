""" create and display example of AutoEncoder model evolution diagram """
from PIL import Image
from random import randrange

# generate a random index of image from which to create evolution diagram
example = randrange(100)

# original image
image = Image.open('original/' + str(example) + '.png')
# image encoded with the digit 0
encoded0 = Image.open('encoded0/' + str(example) + '.png')
# image encoded with the digit 1
encoded1 = Image.open('encoded1/' + str(example) + '.png')
# image auto encoded with the digit 0
auto_encoded0 = Image.open('auto_encoded0/' + str(example) + '.png')
# image auto encoded with the digit 1
auto_encoded1 = Image.open('auto_encoded1/' + str(example) + '.png')

# creates an Image object for evolution diagram
display = Image.new('RGB', (image.width * 3, image.height * 2))
# add original image to display
display.paste(image, (0, image.height // 2))
# adds encoded images to display
display.paste(encoded0, (image.width, 0))
display.paste(encoded1, (image.width, image.height))
# add auto encoded images to display
display.paste(auto_encoded0, (image.width * 2, 0))
display.paste(auto_encoded1, (image.width * 2, image.height))

# display evolution diagram
display.show()
