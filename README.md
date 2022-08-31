<h1 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">Conformal Prediction</h1>
<h3 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">rigorous uncertainty quantification for any machine learning task</h1>

<p align="center">
    <a style="text-decoration:none !important;" href="https://arxiv.org/abs/2107.07511" alt="arXiv"> <img src="https://img.shields.io/badge/paper-arXiv-red" /> </a>
    <a style="text-decoration:none !important;" href="https://people.eecs.berkeley.edu/%7Eangelopoulos/blog/gentle-intro" alt="website"> <img src="https://img.shields.io/badge/website-Berkeley-yellow" /> </a>
    <a style="text-decoration:none !important;" href="https://docs.conda.io/en/latest/miniconda.html" alt="package management"> <img src="https://img.shields.io/badge/conda-env-green" /> </a>
    <a style="text-decoration:none !important;" href="https://opensource.org/licenses/MIT" alt="License"> <img src="https://img.shields.io/badge/license-MIT-blue.svg" /> </a>
    <a style="text-decoration:none !important;" href="https://arxiv.org/abs/2107.07511" alt="arXiv"> <img src="https://img.shields.io/youtube/views/nql000Lu_iE?style=social" /> </a>
    <a style="text-decoration:none !important;" href="https://twitter.com/ml_angelopoulos?ref_src=twsrc%5Etfw" alt="package management"> <img src="https://img.shields.io/twitter/follow/ml_angelopoulos?style=social" /> </a>
</p>

<p>
This repository is the easiest way to start using conformal prediction on real data.
Each notebook in `notebooks` applies conformal prediction to a real prediction problem with a state-of-the-art machine learning model.
</p>

<p>
<b>No need to download the model or data!</b> Raw model outputs on the validation dataset and a small amount of sample data are downloaded automatically by the notebooks. Click on a notebook to see the expected output. You can use these notebooks to experiment with existing methods or as templates to develop your own.
For example,
</p>
<ul>
    <li>[`notebooks/imagenet-smallest-sets.ipynb`](https://github.com/aangelopoulos/conformal-prediction): Imagenet classification with a ResNet152 classifier. Prediction sets guaranteed to contain the true class with 90% probability.</li>
    <li>[`notebooks/meps-cqr.ipynb`](https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/meps-cqr.ipynb): Medical expenditure regression with a Gradient Boosting Regressor and conformalized quantile regression. Prediction intervals guaranteed to contain the true dollar value with 90% probability.</li>
    <li>[`notebooks/multilabel-classification-mscoco.ipynb`](https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/multilabel-classification-mscoco.ipynb): Multilabel image classification on the Microsoft Common Objects in Context (MS-COCO) dataset. Set-valued prediction is guaranteed to contain 90% of the ground truth classes.</li>
    <li>[`notebooks/toxic-text-outlier-detection.ipynb`](https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/toxic-text-outlier-detection.ipynb): Detecting toxic or hateful online comments via conformal outlier detection. No more than 10% of in-distribution data will get flagged as toxic.</li>
    <li>[`notebooks/tumor-segmentation.ipynb`](https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/tumor-segmentation.ipynb): Segmenting gut polyps from endoscopy images. Segmentation masks contain 90% of the ground truth tumor pixels.</li>
    <li>[`notebooks/weather-time-series-distribution-shift`](https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/weather-time-series-distribution-shift.ipynb): Predicting future temperatures around the world using time-series data and weighted conformal prediction. Prediction intervals contaion 90% of true temperatures.</li>
    <li>[`notebooks/imagenet-selective-classification.ipynb`](https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/imagenet-selective-classification.ipynb): When the Imagenet classifier is unsure, it will abstain. Otherwise, it will have an accuracy of 90%, even though the base model was only 77% accurate.</li>
    <li>...and more!</li>
</ul>

<p>
This repository is meant to accompany our paper, the [Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/abs/2107.07511).
See that paper for a detailed explanation of each example, cross-referencing this code.
If you find the repository useful, please cite the paper.
</p>
