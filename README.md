# Quantitative assessment of clearing efficacy

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dbekat/clearing-efficacy/main?urlpath=%2Fdoc%2Ftree%2Funified_analysis.ipynb)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-31311/)
![Commit activity](https://img.shields.io/github/commit-activity/y/dbekat/clearing-efficacy?style=plastic)
![GitHub](https://img.shields.io/github/license/dbekat/clearing-efficacy?color=green&style=plastic)

## Overview

This code is designed as a semi-automated pipeline to assess key parameters of a 3D image, to compare different tissue clearing protocols and choose the most suitable protocol on a case-by-case basis.

### Why?

* There are countless clearing protocols, with the rate of new protocols being discovered rapidly increasing in the last few decades.
* Each has their unique advantages and disadvantages, and are more well suited to different samples and fluorophores
* There is no 'superior' option, and so for a new sample type or fluorophore, it is best practise to determine what tissue clearing protocol can best provide your intended results. 


## How To Run the Code in This Repo

A step-by-step guide is presented below. **You only need to perform steps 1 and 2 once.** Every subsequent time you want to run the code, skip straight to step 3.

### Step 1
#### Install a Python Distribution

We recommend using conda as it's relatively straightforward and makes the management of different Python environments simple. You can install conda from [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation) (miniconda will suffice).

### Step 2
#### Set Up Environment

Once conda is installed, open a terminal (Mac) or command line (Windows) and run the following series of commands:

```
conda create --name clearing-efficacy pip python=3.11
conda activate clearing-efficacy
python -m pip install -r <path to this repo>/requirements.txt
```
where you need to replace `<path to this repo>` with the location on your file system where you downloaded this repo. You will be presented with a list of packages to be downloaded and installed. The following prompt will appear:
```
Proceed ([y]/n)?
```
Hit Enter and all necessary packages will be downloaded and installed - this may take some time. When complete, you can deactivate the environment you have created with the following command.

```
conda deactivate
```
You have successfully set up an environment!

### Step 3
#### Open the notebook

The following commands will launch a Jupyter notebook:
```
conda activate clearing-efficacy
jupyter notebook <path to this repo>/zebrafish_age_estimator.ipynb
```

The Jupyter Notebook should open in your browser - follow the step-by-step instructions in the notebook to run the code. If you are not familiar with Jupyter Notebooks, you can find a detailed introduction [here](https://jupyter-notebook.readthedocs.io/en/latest/notebook.html#introduction).

### (Optional) Step 4
#### Set up your repo to run on Binder

[Binder](https://mybinder.org/) is a really nice way to allow people to run your Jupyter notebooks directly from GitHub - just [follow this handy guide from the Turing Institute](https://the-turing-way.netlify.app/communication/binder/zero-to-binder.html) to get your repo set up.
