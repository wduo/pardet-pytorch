import torch


def single_gpu_test(model, data_loader):
    model.eval()
    probs = []
    gt_labels = []
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)
        probs.append(result['prob'])
        gt_labels.append(result['gt_label'])

    results = dict(probs=probs, gt_labels=gt_labels)
    return results
