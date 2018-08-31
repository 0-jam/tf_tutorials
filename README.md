# TensorFlow Tutorials

TensorFlow tutorials from [official website](https://www.tensorflow.org/)

---

1. [Tested environment](#tested-environment)
    1. [Software](#software)
    1. [Hardware](#hardware)
1. [Todo](#todo)
1. [Installation](#installation)
1. [Troubleshooting](#troubleshooting)

---

## Tested environment

### Software

- Python 3.6.6 on Miniconda 4.5.4
- TensorFlow 1.10.0
- Ubuntu 18.04.1 on Windows Subsystem for Linux (Windows 10 1803 (April 2018))

### Hardware

- [Intel Core i5 7200U](https://ark.intel.com/products/95443/Intel-Core-i5-7200U-Processor-3M-Cache-up-to-3_10-GHz) CPU
- 8GB RAM

## Todo

- [x] Learn and Use ML section
    - [x] [Basic Classification](https://www.tensorflow.org/tutorials/keras/basic_classification)
    - [x] [Text Classification](https://www.tensorflow.org/tutorials/keras/basic_text_classification)
    - [x] [Regression](https://www.tensorflow.org/tutorials/keras/basic_regression)
    - [x] [Overfitting and Underfitting](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit)
    - [x] [Save and Restore Models](https://www.tensorflow.org/tutorials/keras/save_and_restore_models)
- [ ] Translate all comment to Japanese
- [ ] Windows installation instruction

## Installation

```bash
## Clone pyenv repository
$ git clone https://github.com/pyenv/pyenv.git ~/.pyenv

## Set & source environment valiable
$ echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
$ echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
$ echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bashrc
$ source ~/.bashrc

## Install Python (Miniconda)
# TensorFlow for Python 3.7 is unavailable (2018/8/28)
$ pyenv install miniconda3-latest
$ pyenv global miniconda3-latest
# Make sure that Python is successfully installed
$ python -V
Python 3.6.6 :: Anaconda, Inc.

## Update Miniconda packages
$ conda update --all

## Install TensorFlow and other required packages
$ conda install tensorflow numpy matplotlib pandas h5py pyyaml
# Make sure that TensorFlow is successfully installed
$ python -c "import tensorflow as tf; print(tf.__version__)"
1.10.0
```

## Troubleshooting

- If you got `ImportError("Failed to import any qt binding")` while running program that uses Matplotlib, install `pyqt`
