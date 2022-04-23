import cv2


def imread_rgb(fpath):
    img = cv2.imread(fpath, cv2.IMREAD_COLOR)
    if img == None:
        return img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
