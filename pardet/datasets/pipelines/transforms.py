import torchvision.transforms as T

from ..builder import PIPELINES


@PIPELINES.register_module()
class Resize(T.Resize):
    pass


@PIPELINES.register_module()
class Pad(T.Pad):
    pass


@PIPELINES.register_module()
class RandomCrop(T.RandomCrop):
    pass


@PIPELINES.register_module()
class RandomHorizontalFlip(T.RandomHorizontalFlip):
    pass


@PIPELINES.register_module()
class ToTensor(T.ToTensor):
    pass


@PIPELINES.register_module()
class Normalize(T.Normalize):
    pass
