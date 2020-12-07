import os
from PIL import Image, ImageFont, ImageDraw
import glob
from tqdm import tqdm
import pdb

from mmdet.apis import init_detector, inference_detector
import mmcv

import torch
import torchvision.transforms as T

from models.base_block import FeatClassifier, BaseClassifier
from models.resnet import resnet50


def model_init_mmdet():
    config_file = 'configs/faster_rcnn/faster_rcnn_x101_32x4d_fpn_1x_coco.py'
    checkpoint_file = '/mmdet_ckpt/faster_rcnn_x101_32x4d_fpn_1x_coco_20200203-cff10310.pth'
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    return model


def demo_mmdet(model, img_path):
    img = mmcv.imread(img_path)
    result = inference_detector(model, img)
    return result


def model_init_par():
    # model
    backbone = resnet50()
    classifier = BaseClassifier(nattr=113)
    model = FeatClassifier(backbone, classifier)

    # load
    checkpoint = torch.load('/ckpt/ckpt_max.pth', map_location='cpu')
    # unfolded load
    # state_dict = checkpoint['state_dicts']
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]
    #     new_state_dict[name] = v
    # model.load_state_dict(new_state_dict)
    # one-liner load
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dicts'].items()})
    # cuda eval
    model.cuda()
    model.eval()

    # valid_transform
    height, width = 256, 192
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    valid_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])
    return model, valid_transform


def demo_par(model, valid_transform, img):
    # load one image
    img_trans = valid_transform(img)
    imgs = torch.unsqueeze(img_trans, dim=0)
    imgs = imgs.cuda()
    valid_logits = model(imgs)
    valid_probs = torch.sigmoid(valid_logits)
    score = valid_probs.data.cpu().numpy()

    # show the score in the image
    txt_res = []
    for idx in range(len(dt200k_values)):
        if score[0, idx] >= 0.5:
            txt = '%s: %.2f' % (dt200k_values[idx], score[0, idx])
            txt_res.append(txt)

    return txt_res


if __name__ == '__main__':

    # model
    model_mmdet = model_init_mmdet()
    model_par, valid_transform = model_init_par()

    # imgs
    root_dir = '/par_data/imgs/'
    img_paths = glob.glob(os.path.join(root_dir, '**', '*.[pj][np][g]'), recursive=True)

    # L
    for img_path in tqdm(img_paths):
        # mmdet
        result = demo_mmdet(model_mmdet, img_path)
        person_bboxes = result[0]
        person_bboxes_list = []
        for ii in range(person_bboxes.shape[0]):
            x1, y1, x2, y2, score = person_bboxes[ii][0], person_bboxes[ii][1], \
                                    person_bboxes[ii][2], person_bboxes[ii][3], person_bboxes[ii][4]
            if score > 0.6:
                person_bboxes_list.append([x1, y1, x2, y2, score])

        # par
        img = Image.open(img_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        for person_bbox in person_bboxes_list:
            #
            x1, y1, x2, y2, score = person_bbox
            crop_img = img.crop(person_bbox[:-1])
            res_txt = demo_par(model_par, valid_transform, crop_img)

            #
            draw.rectangle(xy=person_bbox[:-1], outline='red', width=1)
            font = ImageFont.truetype('/par/arial.ttf', size=20)
            draw.text((x1, y1 - 20), str(score), (255, 0, 0), font=font)
            positive_cnt = 0
            for txt in res_txt:
                if 'top:' in txt or 'mask:' in txt:
                    draw.text((x1, y1 + 20 * positive_cnt), txt, (255, 0, 0), font=font)
                    positive_cnt += 1

        img.save(os.path.join('/par_data/res/', os.path.basename(img_path)))

        # pdb.set_trace()

        pass
