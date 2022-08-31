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
Each of the <code>notebooks</code> applies conformal prediction to a real prediction problem with a state-of-the-art machine learning model.
</p>

<p align="center"> <b>No need to download the model or data in order to run conformal prediction!</b></p>
<p>
Raw model outputs on the validation dataset and a small amount of sample data are downloaded automatically by the notebooks. Click on a notebook to see the expected output. You can use these notebooks to experiment with existing methods or as templates to develop your own.
</p>
<h3 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">Example notebooks</h1>
<ul>
    <li><a href="https://github.com/aangelopoulos/conformal-prediction"><code>notebooks/imagenet-smallest-sets.ipynb</code></a>: Imagenet classification with a ResNet152 classifier. Prediction sets guaranteed to contain the true class with 90% probability.</li>
    <li><a href="https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/meps-cqr.ipynb"><code>notebooks/meps-cqr.ipynb</code></a>: Medical expenditure regression with a Gradient Boosting Regressor and conformalized quantile regression. Prediction intervals guaranteed to contain the true dollar value with 90% probability.</li>
    <li><a href="https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/multilabel-classification-mscoco.ipynb"><code>notebooks/multilabel-classification-mscoco.ipynb</code></a>: Multilabel image classification on the Microsoft Common Objects in Context (MS-COCO) dataset. Set-valued prediction is guaranteed to contain 90% of the ground truth classes.</li>
    <li><a href="https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/toxic-text-outlier-detection.ipynb"><code>notebooks/toxic-text-outlier-detection.ipynb</code>: Detecting toxic or hateful online comments via conformal outlier detection. No more than 10% of in-distribution data will get flagged as toxic.</li>
    <li><a href="https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/tumor-segmentation.ipynb"><code>notebooks/tumor-segmentation.ipynb</code></a>: Segmenting gut polyps from endoscopy images. Segmentation masks contain 90% of the ground truth tumor pixels.</li>
    <li><a href="https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/weather-time-series-distribution-shift.ipynb"><code>notebooks/weather-time-series-distribution-shift</code></a>: Predicting future temperatures around the world using time-series data and weighted conformal prediction. Prediction intervals contaion 90% of true temperatures.</li>
    <li><a href="https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/imagenet-selective-classification.ipynb"><code>notebooks/imagenet-selective-classification.ipynb</code></a>: When the Imagenet classifier is unsure, it will abstain. Otherwise, it will have an accuracy of 90%, even though the base model was only 77% accurate.</li>
    <li>...and more!</li>
</ul>

<h3 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">Citation</h1>
<p>
This repository is meant to accompany our paper, the [Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/abs/2107.07511).
See that paper for a detailed explanation of each example, cross-referencing this code.
If you find the repository useful, please cite us as follows.
<code>
@article{angelopoulos2021gentle,
  title={A gentle introduction to conformal prediction and distribution-free uncertainty quantification},
  author={Angelopoulos, Anastasios N and Bates, Stephen},
  journal={arXiv preprint arXiv:2107.07511},
  year={2021}
}
</code>
</p>
