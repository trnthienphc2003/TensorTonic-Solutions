import numpy as np

def random_crop(image: np.ndarray, crop_size: int = 224, crop_y: int = None, crop_x: int = None) -> np.ndarray:
    """
    Extract a crop from the image at (crop_y, crop_x). If not given, choose randomly.
    """
    # YOUR CODE HERE
    H, W, C = image.shape
    if crop_y is None:
        crop_y = np.random.randint(0, H - crop_size + 1)
    if crop_x is None:
        crop_x = np.random.randint(0, W - crop_size + 1)
    return image[crop_y : crop_y + crop_size, crop_x : crop_x + crop_size, :]

def random_horizontal_flip(image: np.ndarray, p: float = 0.5, flip_rand: float = None) -> np.ndarray:
    """
    Flip image horizontally if flip_rand < p. If flip_rand not given, generate randomly.
    """
    # YOUR CODE HERE
    if flip_rand is None:
        flip = np.random.random()

    if flip_rand < p:
        image = image[:, ::-1, :]
    return image