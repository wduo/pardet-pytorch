from abc import ABCMeta, abstractmethod
import torch.nn as nn


class BaseBody(nn.Module, metaclass=ABCMeta):
    """Base class for body detectors."""

    def __init__(self):
        super(BaseBody, self).__init__()

    @abstractmethod
    def extract_feat(self, imgs):
        """Extract features from images."""
        pass
