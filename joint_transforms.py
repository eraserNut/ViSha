import random

from PIL import Image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, manual_random=None):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask, manual_random)
        return img, mask


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask, manual_random=None):
        if manual_random is None:
            if random.random() < 0.5:
                return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
            return img, mask
        else:
            if manual_random < 0.5:
                return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
            return img, mask


class Resize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask, manual_random=None):
        assert img.size == mask.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)
