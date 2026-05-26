Steps I had to take to install:

* git clone
* created a virtual environment with Python 3.11
* installed the requirements
```sh
git lfs install
git lfs clone https://huggingface.co/WangZeJun/simbert-base-chinese WangZeJun/simbert-base-chinese
```

```sh
git clone https://www.modelscope.cn/syq163/outputs.git
```

```sh
.venv ❯ python
Python 3.11.4 (v3.11.4:d2340ef257, Jun  6 2023, 19:15:51) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import nltk
>>> nltk.download('averaged_perceptron_tagger_eng')
[nltk_data] Downloading package averaged_perceptron_tagger_eng to
[nltk_data]     /Users/corey/nltk_data...
[nltk_data]   Unzipping taggers/averaged_perceptron_tagger_eng.zip.
True
```
* pip install unidecode
* (might take this off if I don't use it) pip install num2words
* pip install ollama