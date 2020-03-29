from PIL import Image
from random import randrange

example = randrange(100)

image = Image.open('original/' + str(example) + '.png')
encoded0 = Image.open('encoded0/' + str(example) + '.png')
encoded1 = Image.open('encoded1/' + str(example) + '.png')
auto_encoded0 = Image.open('auto_encoded0/' + str(example) + '.png')
auto_encoded1 = Image.open('auto_encoded1/' + str(example) + '.png')

display = Image.new('RGB', (image.width * 3, image.height * 2))
display.paste(image, (0, image.height // 2))
display.paste(encoded0, (image.width, 0))
display.paste(encoded1, (image.width, image.height))
display.paste(auto_encoded0, (image.width * 2, 0))
display.paste(auto_encoded1, (image.width * 2, image.height))

display.show()
