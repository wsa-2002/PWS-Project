import cv2


def are_different_images(img_1, img_2, threshold = 4) -> bool:
    """
    return: True for images are different.
    """
    hash1 = dhash(img_1)
    hash2 = dhash(img_2)

    difference = bin(hash1 ^ hash2).count('1')
    if difference > threshold:
        return True
    return False


def dhash(image, hash_size=8):

    resized = cv2.resize(image, (hash_size + 1, hash_size))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    diff = gray[:, 1:] > gray[:, :-1]

    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
