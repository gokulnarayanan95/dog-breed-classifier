from keras.preprocessing import image


def read_image(path, size):
    img = image.load_img(path, False, size)
    img = image.img_to_array(img)
    return img
