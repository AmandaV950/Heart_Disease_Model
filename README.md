# Heart Disease Machine Learning model

***Amanda Vel, 2025***

A machine learning classification model built for predicting heart failure from a set of clinical data. Three models (logistic regression, random forest, K nearest neighbors) were trained, fitted and evaluated. K nearest neighbors was the most accurate, with a 5-fold VC showing a mean accuracy of 86%.

## Table of Contents
- [Files](#Files)
- [Dataset](#Dataset)
- [Setup](#Setup)
- [License](#License)

## Files
- `heart.csv` - Dataset used in analysis
- `code.ipynb` - Script as an Interactive Python Notebook. End-to-end from importing data, preprocessing, model creations and predictions.
- `code.py` - Plain text python script of source code

>[!NOTE]
> .py file supplided for convenience, but reccomended to use the .ipynb file for running code in JupyterLab

## Dataset
The heart failure dataset was obtained from Kaggle ([link](https://www.kaggle.com/fedesoriano/heart-failure-prediction)) and contained 11 features over 918 observations.

<img width="467" height="314" alt="image" src="https://github.com/user-attachments/assets/cdd64e72-9838-4405-8412-eb1b2746c595" />


## Setup
**Prerequisities:**
- Python 3.6+
- JupyterLab environment
- scikit-learn, pandas, seaborn, & numpy packages

**Install instructions:**

Install UV:
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
Open a Terminal, create a folder and move into project directory:
```
mkdir my_project
cd my_project
```
Create a virtual environment and install packages:
```
uv init --bare
uv add jupyterlab
uv add scikit-learn pandas seaborn numpy
```

## Usage
Download the script (`code.ipynb`) and dataset (`heart.csv`) into this directory.

From the directory, open jupyter lab:
```
uv run jupyter lab
```
Finally, run the provided script.
## License
This project is licensed under the [BSD-3-Clause license](LICENSE)
