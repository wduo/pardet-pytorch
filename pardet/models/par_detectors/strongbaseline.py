from collections import OrderedDict

import torch
import torch.nn as nn

from ..builder import CLASSIFIERS, PARNETS, build_bockbone, build_classifier, build_loss


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

    def __init__(self, backbone, classifier, loss):
        super(StrongBaseline, self).__init__()

        self.backbone = build_bockbone(backbone)
        self.classifier = build_classifier(classifier)
        self.loss = build_loss(loss)

    def fresh_params(self):
        params = self.classifier.fresh_params()
        return params

    def finetune_params(self):
        return self.backbone.parameters()

    def extract_feat(self, img):
        x = self.backbone(img)
        return x

    def forward_train(self, **kwargs):
        feat_map = self.extract_feat(kwargs['img'].cuda())
        logits = self.classifier(feat_map)

        losses = dict()
        loss = self.loss(logits, kwargs['gt_label'].cuda(), kwargs['weights'])
        losses.update(loss)

        return losses

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def train_step(self, data, optimizer, **kwargs):
        losses = self(**data, **kwargs)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_name']))
        return outputs

    def val_step(self, data, optimizer, **kwargs):
        losses = self(**data, **kwargs)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_name']))
        return outputs

    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def forward_test(self, **kwargs):
        imgs = kwargs.pop('img')
        num_augs = len(imgs)
        if num_augs == 1:
            return self.simple_test(imgs, **kwargs)
        else:
            return self.aug_test(imgs, **kwargs)

    def simple_test(self, img, **kwargs):
        feat_map = self.extract_feat(img.cuda())
        logit = self.classifier(feat_map)
        prob = torch.sigmoid(logit).detach().cpu().numpy()

        gt_label = kwargs['gt_label'].detach().cpu().numpy()
        result = dict(prob=prob, gt_label=gt_label)
        return result

    def aug_test(self, imgs, **kwargs):
        # TODO: support test augmentation for predefined proposals
        pass
