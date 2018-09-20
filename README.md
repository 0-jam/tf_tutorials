# TensorFlow Tutorials

- TensorFlow tutorials from [official website](https://www.tensorflow.org/) + a few modification
- The main purpose of this repository is learning how to use TensorFlow and how machine learning works

---

1. [Environment](#environment)
    1. [Software](#software)
    1. [Hardware](#hardware)
1. [Todo](#todo)
1. [Installation](#installation)
    1. [Install Python with Miniconda](#install-python-with-miniconda)
    1. [Install Python without Miniconda](#install-python-without-miniconda)
    1. [Install TensorFlow and other required packages](#install-tensorflow-and-other-required-packages)
1. [Note](#note)
1. [Troubleshooting](#troubleshooting)

---

## Environment

### Software

- Python 3.6.6 on Miniconda 4.5.4
- Python 3.6.6 without Miniconda
- TensorFlow 1.10.0, 1.10.1
- Ubuntu 18.04.1 on Windows Subsystem for Linux (Windows 10 Home 1803 (April 2018))

### Hardware

- CPU: Intel [Core i5 7200U](https://ark.intel.com/products/95443/Intel-Core-i5-7200U-Processor-3M-Cache-up-to-3_10-GHz)
- RAM: 8GB

## Todo

- [x] Learn and Use ML section
    - [x] [Basic classification](https://www.tensorflow.org/tutorials/keras/basic_classification)
    - [x] [Text classification](https://www.tensorflow.org/tutorials/keras/basic_text_classification)
    - [x] [Regression](https://www.tensorflow.org/tutorials/keras/basic_regression)
    - [x] [Overfitting and underfitting](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit)
    - [x] [Save and restore models](https://www.tensorflow.org/tutorials/keras/save_and_restore_models)
- [ ] Run these scripts on my own desktop PC
    - CPU: AMD [Ryzen 7 1700](https://www.amd.com/ja/products/cpu/amd-ryzen-7-1700)
    - RAM: 16GB
    - OS: Ubuntu 18.04.1 on Windows Subsystem for Linux (Windows 10 Home 1803 (April 2018))
- [ ] Try Arch Linux (for my personal interest)
- [ ] Try CUDA
    - Currently I cannot try this because I don't have NVIDIA GPU :(
- Try another tutorials
    - Research and experimentation
        - [x] [Eager execution basics](https://www.tensorflow.org/tutorials/eager/eager_basics)
        - [x] [Automatic differentiation and gradient tape](https://www.tensorflow.org/tutorials/eager/automatic_differentiation)
        - [x] [Custom training: basics](https://www.tensorflow.org/tutorials/eager/custom_training)
        - [ ] [Custom layers](https://www.tensorflow.org/tutorials/eager/custom_layers)
        - [ ] [Custom training: walkthrough](https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough)
    - Genarative models
        - [x] [Text generation](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/eager/python/examples/generative_examples/text_generation.ipynb)
    - Non-ML
        - [x] [Mandelbrot set](https://www.tensorflow.org/tutorials/non-ml/mandelbrot)
- [ ] Refactoring
- [ ] Translate all comments and this README to Japanese
    - Translated files should be renamed such as 'basic_classification.ja.py' ...
- [ ] Windows installation instruction

## Installation

```bash
## Clone pyenv repository
$ git clone https://github.com/pyenv/pyenv.git ~/.pyenv

## Set & source environment variable
$ echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
$ echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
$ echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bashrc
$ source ~/.bashrc
```

### Install Python with Miniconda

```bash
$ pyenv install miniconda3-latest
$ pyenv global miniconda3-latest

# Make sure that Python is successfully installed
$ python -V
Python 3.6.6 :: Anaconda, Inc.
# Update Miniconda packages
$ conda update --all
```

### Install Python without Miniconda

```bash
# Before installing Python, install tk-dev for matplotlib dependency
# to avoid "ModuleNotFoundError: No module named '_tkinter'"
$ sudo apt install tk-dev
# Build Python (tk-enabled)
# TensorFlow for Python 3.7.x is unavailable (2018/9/18)
$ pyenv install 3.6.6
$ pyenv global 3.6.6
# Make sure that Python is successfully installed
$ python -V
Python 3.6.6
# Update pip packages
$ pip install -U pip
$ pip install pip-review
$ pip-review --auto
```

### Install TensorFlow and other required packages

- If you have installed Python without Miniconda, replace `conda` with `pip`

```bash
$ conda install tensorflow numpy matplotlib pandas h5py pyyaml
# "Generative Model" section
$ conda install unidecode
# "Non-ML" section
$ conda install ipython pillow
# Make sure that TensorFlow is successfully installed
$ python -c "import tensorflow as tf; print(tf.__version__)"
1.10.1
```

## Note

- All opening window function (such as `plt.show()`) was replaced by saving as an image (such as `plt.savefig('filename')`)
    - `show()` method does not open window in WSL

## Troubleshooting

- If you got `ImportError("Failed to import any qt binding")` while running script that uses Matplotlib, install `pyqt`
