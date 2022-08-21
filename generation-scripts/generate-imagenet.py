import os, shutil
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt 
import random
import pdb
from tqdm import tqdm
from pathlib import Path
ABSPATH = Path(__file__).parent.absolute()

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

if __name__ == "__main__":
    with torch.no_grad():
        # Transform as in https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L92
        transform = torchvision.transforms.Compose([
                        torchvision.transforms.Resize(256),
                        torchvision.transforms.CenterCrop(224),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std= [0.229, 0.224, 0.225])
                    ])

        # Get the Imagenet dataset
        imagenet_dataset = ImageFolderWithPaths('~/data/ilsvrc/val', transform)

        # Initialize loaders
        imagenet_loader = torch.utils.data.DataLoader(imagenet_dataset, batch_size=512, shuffle=False, pin_memory=True)

        # Get the model
        model = torchvision.models.resnet152(pretrained=True,progress=True).cuda()
        model.eval()
        scores = np.ones((len(imagenet_dataset),1000))
        labels = np.ones((len(imagenet_dataset),))
        example_indexes = np.random.choice(len(imagenet_dataset), size=(500,), replace=False, p=None) # Randomly sample 500 images for people to play with
        paths = []
        counter = 0

        print("computing the scores")
        for batch in tqdm(imagenet_loader):
            scores[counter:counter+batch[0].shape[0],:] = model(batch[0].cuda()).softmax(dim=1).cpu().numpy()
            labels[counter:counter+batch[1].shape[0]] = batch[1].numpy().astype(int)
            paths += batch[2]
            counter += batch[0].shape[0]

        print("saving the scores and labels")
        os.makedirs(str(ABSPATH.parent) + '/data/imagenet/examples', exist_ok=True)
        np.savez(str(ABSPATH.parent) + '/data/imagenet/imagenet-resnet152.npz', smx=scores, labels=labels, example_indexes=example_indexes)

        print("moving a subset of images to the examples folder")
        for idx in example_indexes:
            shutil.copy(paths[idx], str(ABSPATH.parent) + '/data/imagenet/examples/' + str(idx) + '.jpeg')
