import torch.nn as nn

from ..builder import CLASSIFIERS, PARNETS, build_bockbone, build_classifier


@CLASSIFIERS.register_module()
class BaseClassifier(nn.Module):
    def __init__(self, nattr):
        super().__init__()
        self.logits = nn.Sequential(
            nn.Linear(2048, nattr),
            nn.BatchNorm1d(nattr)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def fresh_params(self):
        return self.parameters()

    def forward(self, feature):
        feat = self.avg_pool(feature).view(feature.size(0), -1)
        x = self.logits(feat)
        return x


@PARNETS.register_module()
class StrongBaseline(nn.Module):

    def __init__(self, backbone, classifier):
        super(StrongBaseline, self).__init__()

        self.backbone = build_bockbone(backbone)
        self.classifier = build_classifier(classifier)

    def fresh_params(self):
        params = self.classifier.fresh_params()
        return params

    def finetune_params(self):
        return self.backbone.parameters()

    def forward(self, x, label=None):
        feat_map = self.backbone(x)
        logits = self.classifier(feat_map)
        return logits
