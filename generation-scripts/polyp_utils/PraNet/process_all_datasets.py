import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from lib.PraNet_Res2Net import PraNet
from utils.dataloader import test_dataset
import pdb

for _data_name in ['HyperKvasir', 'CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
    data_path = './data/TestDataset/{}/'.format(_data_name)
    save_path = './results/PraNet/{}/'.format(_data_name)
    model_path = './snapshots/PraNet_Res2Net/PraNet-19.pth'
    batch_size = 256

    model = PraNet()
    model.load_state_dict(torch.load())
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, batch_size)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        print(f"\33[2K\r Processing {name}", end="")
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res5, res4, res3, res2 = model(image)
        res = res2
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # Save the image as a float numpy array, since otherwise it will be quantized to 256 levels.
        np.save(save_path+name,res)
