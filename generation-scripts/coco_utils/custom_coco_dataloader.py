import os.path
from typing import Any, Callable, Optional, Tuple, List

from PIL import Image

import torch
from torchvision.datasets.vision import VisionDataset

import pdb

def get_correspondences(model_arr,dset_dict):
    corr = {}
    for i in range(model_arr.shape[0]):
        corr[i] = list(dset_dict.keys())[i]
    corr = {y:x for x,y in corr.items()}
    return corr

class CocoDetectionWithPaths(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    This version was modified by @aangelopoulos to also output file paths and be compatible with PyTorch dataloader

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        classes_list,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.corr = get_correspondences(classes_list,self.coco.cats)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        label = torch.zeros((80,))

        if len(target) != 0:
            annotations = self.coco.getAnnIds(imgIds=int(target[0]['image_id']))

            for annotation in self.coco.loadAnns(annotations):
                label[self.corr[annotation['category_id']]] = 1

        return image, label, self.coco.loadImgs(id)[0]["file_name"]


    def __len__(self) -> int:
        return len(self.ids)
