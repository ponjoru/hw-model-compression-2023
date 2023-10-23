import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

from src.utils import compute_pytorch_latency, get_transforms
from src.dataset import WiderFaceDataset
from src.evaluator import WiderFaceEvaluator
from src.upd_scrfd.utils import postprocess
from src.upd_scrfd.scrfd import SCRFD_500M


def eval(model, dataloader, postprocess, device='cpu', debug=False):
    model.eval()
    desc = f'Evaluating'
    bar_format = '{l_bar}{bar}{r_bar}'
    pbar = tqdm(dataloader, bar_format=bar_format, total=len(dataloader), desc=desc)

    gt_dir = '../data/widerface/val/gt'
    evaluator = WiderFaceEvaluator(gt_dir, iou_thresh=0.5)

    with torch.no_grad():
        for i, batch in enumerate(pbar):
            img = batch['image']
            img_shape = batch['image'].size()[-2:]
            targets = {'bb': batch['bb'].to(device), 'kp': batch['kp'].to(device)}
            img = img.to(device)

            raw_output = model(img)
            output = postprocess(raw_output)

            meta_data = {'img_id': batch['img_id'], 'img0_shape': batch['img_shape'], 'img1_shape': img_shape,
                         'img_path': batch['img_path']}

            evaluator.add_batch(output, targets, meta_data)

            if debug and i > 5:
                break

        metrics = evaluator.compute()
    return metrics


if __name__ == '__main__':
    device = 'cpu'
    ds_path = '../data/widerface'
    mode = 'val'

    transforms = get_transforms(mode=mode)
    dataset = WiderFaceDataset(ds_path, mode, min_size=None, transforms=transforms, color_layout='RGB')
    dataloader = DataLoader(dataset, batch_size=8, num_workers=4, pin_memory=True, collate_fn=dataset.collate_fn)
    postproc = lambda x: postprocess(x, conf_thresh=0.02, iou_thresh=0.45)

    # ------------------------- eval -------------------------
    model = SCRFD_500M(nc=1).eval()
    model.load_from_checkpoint('../weights/upd_SCRFD_500M_KPS.pth')
    print('SCRFD_500M_KPS model')

    latency = compute_pytorch_latency(model, (1, 3, 640, 640))
    print(f'latency: {latency:.2f}ms')
    metrics = eval(model, dataloader, postproc, device)
    print(metrics)

    """
    SCRFD_500M_KPS model
    latency: 50.68ms
    Evaluating: 100 % |██████████ | 404 / 404[06:34 < 00:00, 1.02 it / s]
    Detection Metrics:
    easy_AP: 0.9071, medium_AP: 0.8805, hard_AP: 0.6768
    """
