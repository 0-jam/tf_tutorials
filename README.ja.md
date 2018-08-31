# TensorFlowチュートリアル

- TensorFlow[公式サイト](https://www.tensorflow.org/)のチュートリアルに、少しだけ追記・修正したもの
- TensorFlowの使い方と、機械学習の動きを勉強するためのリポジトリ

---

1. [動作確認環境](#動作確認環境)
    1. [ソフトウェア](#ソフトウェア)
    1. [ハードウェア](#ハードウェア)
1. [Todo](#todo)
1. [インストール](#インストール)
1. [トラブルシューティング](#トラブルシューティング)

---

## 動作確認環境

### ソフトウェア

- Miniconda 4.5.4上のPython 3.6.6
- TensorFlow 1.10.0
- Windows Subsystem for Linux (Windows 10 1803 (April 2018))上のUbuntu 18.04.1

### ハードウェア

- [Intel Core i5 7200U](https://ark.intel.com/ja/products/95443/Intel-Core-i5-7200U-Processor-3M-Cache-up-to-3_10-GHz) CPU
- 8GB RAM

## Todo

- [x] Learn and Use ML（機械学習を学ぶ・使う）セクション
    - [x] [Basic Classification](https://www.tensorflow.org/tutorials/keras/basic_classification)（基本的な分類）
    - [x] [Text Classification](https://www.tensorflow.org/tutorials/keras/basic_text_classification)（テキスト分類）
    - [x] [Regression](https://www.tensorflow.org/tutorials/keras/basic_regression)（回帰分析）
    - [x] [Overfitting and Underfitting](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit)（過学習と未学習）
    - [x] [Save and Restore Models](https://www.tensorflow.org/tutorials/keras/save_and_restore_models)（モデルの保存と読み込み）
- [ ] コメントの日本語訳
    - 翻訳版は別ファイルにしたほうがいいかも…
- [ ] Windows上でのインストール手順

## インストール

```bash
## pyenvリポジトリをclone
$ git clone https://github.com/pyenv/pyenv.git ~/.pyenv

## 環境変数設定＆読み込み
$ echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
$ echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
$ echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bashrc
$ source ~/.bashrc

## Python (Miniconda)をインストール
# TensorFlowはPython 3.7非対応 (2018/8/28)
$ pyenv install miniconda3-latest
$ pyenv global miniconda3-latest
# Pythonが正しくインストールされているか確認
$ python -V
Python 3.6.6 :: Anaconda, Inc.

## Minicondaパッケージをアップデート
$ conda update --all

## TensorFlowとその他のここで使うパッケージをインストール
$ conda install tensorflow numpy matplotlib pandas h5py pyyaml
# TensorFlowが正しくインストールされているか確認
$ python -c "import tensorflow as tf; print(tf.__version__)"
1.10.0
```

## トラブルシューティング

- Matplotlibを使うプログラムの実行中に`ImportError("Failed to import any qt binding")`が出たら、`pyqt`をインストール