from ..builder import BODYNET
from .base import BaseBody


@BODYNET.register_module()
class BodyFaster1(BaseBody):
    def __init__(self, name, train_cfg, test_cfg):
        super(BodyFaster1, self).__init__()

        self.name = name
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        pass

    def extract_feat(self, imgs):
        pass
