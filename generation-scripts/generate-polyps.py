import os, sys, shutil
sys.path.insert(1, os.path.join(sys.path[0], './polyp_utils'))
import torch
import numpy as np
from skimage.transform import resize
from pathlib import Path
from PraNet.lib.PraNet_Res2Net import PraNet
from PraNet.utils.dataloader import test_dataset
from tqdm import tqdm

ABSPATH = Path(__file__).parent.absolute()

def get_num_examples(folders):
    num = 0
    for folder in folders:
        num += len([name for name in os.listdir(folder + '/images/')])
    return num

# Computes sigmoid scores and ground truth masks from the model and loader 
def get_sigmoids_targets(model, folders):
    T = 10 
    test_size = 352

    num_examples = get_num_examples(folders)
    print(f'Caching {num_examples} labeled examples.')
    img_paths = ['']*num_examples
    mask_paths = ['']*num_examples
    sigmoids = np.zeros((num_examples, test_size, test_size))
    gt_masks = np.zeros((num_examples, test_size, test_size))
    
    k = 0

    for data_path in folders:
        image_root = data_path + '/images/'
        gt_root = data_path + '/masks/'
        test_loader = test_dataset(image_root, gt_root, test_size)
        print(f"Processing dataset: {data_path}")

        for i in tqdm(range(test_loader.size)):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            res5, res4, res3, res2 = model(image)
            
            # Populate the arrays
            img_paths[k] = image_root + '/' + name
            mask_paths[k] = gt_root + '/' + name
            sigmoids[k,:,:] = (res2/T).sigmoid().detach().cpu().numpy()
            gt_masks[k,:,:] = resize(gt, (test_size, test_size), anti_aliasing=False) > 0.5
            k += 1

    return sigmoids, gt_masks, np.array(img_paths), np.array(mask_paths)

if __name__ == "__main__":
    with torch.no_grad():
        # TODO: Change this dataset path to yours 
        folders = ['HyperKvasir', 'CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
        folders = ['/home/aa/Code/conformal-risk-control/polyps/PraNet/data/TestDataset/' + x for x in folders]
        # Setup model
        model_path = './polyp_utils/PraNet-19.pth' 
        if not os.path.exists(model_path):
            os.system('gdown 1pUE99SUQHTLxS9rabLGe_XTDwfS6wXEw -O ' + model_path)
            print("Model downloaded!")
        model = PraNet()
        model.load_state_dict(torch.load(model_path))
        model.cuda()
        model.eval()

        sigmoids, targets, img_paths, mask_paths = get_sigmoids_targets(model, folders)

        example_indexes = np.random.choice(img_paths.shape[0], size=(500,), replace=False, p=None) # Randomly sample 500 images for people to play with

        print("saving the sigmoid scores and labels")
        os.makedirs(str(ABSPATH.parent) + '/data/polyps/examples', exist_ok=True)
        np.savez(str(ABSPATH.parent) + '/data/polyps/polyps-pranet.npz', sgmd=sigmoids, targets=targets, example_indexes=example_indexes)

        print("moving a subset of images to the examples folder")
        for idx in example_indexes:
            shutil.copy(img_paths[idx], str(ABSPATH.parent) + '/data/polyps/examples/' + str(idx) + '.jpg')
            shutil.copy(mask_paths[idx], str(ABSPATH.parent) + '/data/polyps/examples/' + str(idx) + '_gt_mask.jpg')
