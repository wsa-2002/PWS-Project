import cv2


def are_different_images(img_1, img_2, threshold: float = 0.07) -> bool:
    """
    return: True for images are different.
    """
    hash_1 = a_hash(img_1)
    hash_2 = a_hash(img_2)

    height, width, _ = img_1.shape

    hash_cmp = hash_comparison(hash_1, hash_2) / (height * width)
    if hash_cmp > threshold:
        return True
    return False


def a_hash(img):
    s = 0
    hash_str = ''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    for i in range(height):
        for j in range(width):
            s += gray[i, j]
    avg = s / (height * width)

    for i in range(height):
        for j in range(width):
            hash_str += '1' if gray[i, j] > avg else '0'
    return hash_str


def hash_comparison(hash_1: str, hash_2: str):
    if len(hash_1) != len(hash_2):
        return -1
    n = 0
    for i, _ in enumerate(hash_1):
        if hash_1[i] != hash_2[i]:
            n += 1
    return n
