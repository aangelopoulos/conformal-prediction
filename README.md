<h1 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">Conformal Prediction</h1>
<h3 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">rigorous uncertainty quantification for any machine learning task</h3>

<p align="center">
    <a style="text-decoration:none !important;" href="https://arxiv.org/abs/2107.07511" alt="arXiv"><img src="https://img.shields.io/badge/paper-arXiv-red" /></a>
    <a style="text-decoration:none !important;" href="https://people.eecs.berkeley.edu/%7Eangelopoulos/blog/posts/gentle-intro" alt="website"><img src="https://img.shields.io/badge/website-Berkeley-yellow" /></a>
    <a style="text-decoration:none !important;" href="https://docs.conda.io/en/latest/miniconda.html" alt="package management"> <img src="https://img.shields.io/badge/conda-env-green" /></a>
    <a style="text-decoration:none !important;" href="https://opensource.org/licenses/MIT" alt="License"><img src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
    <a style="text-decoration:none !important;" href="https://www.youtube.com/watch?v=nql000Lu_iE" alt="arXiv"><img src="https://img.shields.io/youtube/views/nql000Lu_iE?style=social" /></a>
    <a style="text-decoration:none !important;" href="https://twitter.com/ml_angelopoulos?ref_src=twsrc%5Etfw" alt="package management"><img src="https://img.shields.io/twitter/follow/ml_angelopoulos?style=social" /></a>
    <a style="text-decoration:none !important;" href="https://twitter.com/stats_stephen" alt="package management"><img src="https://img.shields.io/twitter/follow/stats_stephen?style=social" /></a>
</p>

<p>
This repository is the easiest way to start using conformal prediction (a.k.a. conformal inference) on real data.
Each of the <code>notebooks</code> applies conformal prediction to a real prediction problem with a state-of-the-art machine learning model.
</p>

<p align="center"> <b>No need to download the model or data in order to run conformal</b></p>
<p>
Raw model outputs for several large-scale real-world datasets and a small amount of sample data from each dataset are downloaded automatically by the notebooks. You can develop and test conformal prediction methods entirely in this sandbox, without ever needing to run the original model or download the original data. Open a notebook to see the expected output. You can use these notebooks to experiment with existing methods or as templates to develop your own. 
</p>
<h3 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">Example notebooks</h3>
<ul>
    <li><a href="https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/imagenet-smallest-sets.ipynb"><code>notebooks/imagenet-smallest-sets.ipynb</code></a>: Imagenet classification with a ResNet152 classifier. Prediction sets guaranteed to contain the true class with 90% probability.</li>
    <li><a href="https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/meps-cqr.ipynb"><code>notebooks/meps-cqr.ipynb</code></a>: Medical expenditure regression with a Gradient Boosting Regressor and conformalized quantile regression. Prediction intervals guaranteed to contain the true dollar value with 90% probability.</li>
    <li><a href="https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/multilabel-classification-mscoco.ipynb"><code>notebooks/multilabel-classification-mscoco.ipynb</code></a>: Multilabel image classification on the Microsoft Common Objects in Context (MS-COCO) dataset. Set-valued prediction is guaranteed to contain 90% of the ground truth classes.</li>
    <li><a href="https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/toxic-text-outlier-detection.ipynb"><code>notebooks/toxic-text-outlier-detection.ipynb</code></a>: Detecting toxic or hateful online comments via conformal outlier detection. No more than 10% of in-distribution data will get flagged as toxic.</li>
    <li><a href="https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/tumor-segmentation.ipynb"><code>notebooks/tumor-segmentation.ipynb</code></a>: Segmenting gut polyps from endoscopy images. Segmentation masks contain 90% of the ground truth tumor pixels.</li>
    <li><a href="https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/weather-time-series-distribution-shift.ipynb"><code>notebooks/weather-time-series-distribution-shift</code></a>: Predicting future temperatures around the world using time-series data and weighted conformal prediction. Prediction intervals contaion 90% of true temperatures.</li>
    <li><a href="https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/imagenet-selective-classification.ipynb"><code>notebooks/imagenet-selective-classification.ipynb</code></a>: When the Imagenet classifier is unsure, it will abstain. Otherwise, it will have an accuracy of 90%, even though the base model was only 77% accurate.</li>
    <li>...and more!</li>
</ul>

<h3 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">Notebooks can be run immediately using the provided Google Colab links</h3>
<h5 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">Colab links are in the top cell of each notebook</h5>

<p>
    To run these notebooks locally, you just need to have the correct dependencies installed and press <code>run all cells</code>! The notebooks will automatically download all required data and model outputs.  You will need 1.5GB of space on your computer in order for the notebook to store the auto-downloaded data. If you want to see how we generated the precomputed model outputs and data subsamples, see the files in <code>generation-scripts</code>. There is one for each dataset. To create a <code>conda</code> environment with the correct dependencies, run <code>conda env create -f environment.yml</code>. If you still get a dependency error, make sure to activate the <code>conformal</code> environment within the Jupyter notebook.
</p>

<h3 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">Citation</h3>
<p>
This repository is meant to accompany our paper, the <a href="https://arxiv.org/abs/2107.07511">Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification</a>.
In that paper is a detailed explanation of each example and attributions.
If you find this repository useful, in addition to the relevant methods and datasets, please cite:
</p>
<pre><code>@article{angelopoulos2021gentle,
  title={A gentle introduction to conformal prediction and distribution-free uncertainty quantification},
  author={Angelopoulos, Anastasios N and Bates, Stephen},
  journal={arXiv preprint arXiv:2107.07511},
  year={2021}
}</code></pre>
<h3 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">Videos</h3>
If you're interested in learning about conformal prediction in video form, watch our videos below!

<h4 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">A Tutorial on Conformal Prediction</h4>
<p align="center"><a href="https://www.youtube.com/watch?v=nql000Lu_iE"> <img width="350" src="https://img.youtube.com/vi/nql000Lu_iE/maxresdefault.jpg" /> </a></p>

<h4 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">A Tutorial on Conformal Prediction Part 2: Conditional Coverage</h4>
<p align="center"><a href="https://www.youtube.com/watch?v=TRx4a2u-j7M"> <img width="350" src="https://img.youtube.com/vi/TRx4a2u-j7M/maxresdefault.jpg" /> </a></p>

<h4 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">A Tutorial on Conformal Prediction Part 3: Beyond Conformal Prediction</h4>
<p align="center"><a href="https://www.youtube.com/watch?v=37HKrmA5gJE"> <img width="350" src="https://img.youtube.com/vi/37HKrmA5gJE/maxresdefault.jpg" /> </a></p>
