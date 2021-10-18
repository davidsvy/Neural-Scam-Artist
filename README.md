<h1 align="center">
  <b>Neural Scam Artist</b><br>
</h1>


<p align="center">
      <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/python-3.7-blue.svg" /></a>
       <a href= "https://pytorch.org/">
        <img src="https://img.shields.io/badge/PyTorch-1.9-FF0000.svg" /></a>
       <a href= "https://github.com/davidsvy/Neural-Scam-Artist/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/license-MIT-white.svg" /></a>
</p>

TL;DR\
A dataset of scam emails is scraped from an anti-fraud website. The dataset is then deduplicated
using MinHash and LSH. The deduplicated dataset is used for fine-tuning GPT-2.


<p align="center">
  <img src="https://github.com/davidsvy/Neural-Scam-Artist/blob/master/assets/comic.jpg?raw=true" />
</p>



<p align="center">
  Comic stolen from <a href="https://www.agent-x.com.au/">Agent-X Comics</a>.
  
</p>



:book: Table of contents
===

<!--ts-->
  * [➤ Project Description](#project-description)
    * [➤ Objective](#objective)
    * [➤ Web Scraper](#web-scraper)
    * [➤ Deduplication](#deduplication)
    * [➤ GPT-2](#gpt-2)
  * [➤ Shared Files](#shared-files)
  * [➤ Requirements](#requirements)
  * [➤ Installation](#installation)
  * [➤ Usage](#usage)
<!--te-->

:cloud: Project Description
===

Objective
---

The goal of this project is create a new dataset of fraudulent emails that can advance the
research on intelligent email assistants.

Web Scraper
---

Data is scraped from the website [https://antifraudintl.org/](https://antifraudintl.org/). 
At first, a set of thread urls is collected and stored. Then, each thread is searched for 
emails. For each thread, at most one email is kept as the rest are duplicates. Metadata 
(Subject, Date etc) is removed. The resultant dataset is stored inside a csv file.

Deduplication
---
To avoid the quadratic complexity, a cheap alternative is selected: MinHash and LSH using the [datasketch library](https://github.com/ekzhu/datasketch). For each document, this method 
efficiently locates its nearest neighbors. Because this leads to a a large amount of false
negatives (i.e. dulpicate documents that are classified as non-duplicates), the approach is
extended by creating a duplicate graph. Nodes in this graph represent documents and are connected
with an edge if their respective documents have been classified as duplicates. To deduplicate the 
dataset, [connected components](https://en.wikipedia.org/wiki/Component_(graph_theory)) of the 
graph are located and for each component only a single node is selected. A 
[readability criterion](https://en.wikipedia.org/wiki/Readability) is used for selection.

GPT-2
---

A small pretrained GPT-2 model from the 
[Huggingface library](https://huggingface.co/transformers/model_doc/gpt2.html#gpt2lmheadmodel)
is fine-tuned on the deduplicated dataset. A collection of ~~cherry-picked~~ randomly selected 
generated samples can be found here [here](https://github.com/davidsvy/Neural-Scam-Artist/blob/main/generated_samples/generated_samples.txt).

:file_folder: Shared Files
===


| Resource | Size | #Samples | Link |
|-------------------|---|---|---|
| **Full dataset**          | 128.5 MB  | 85,160  | [Link](https://drive.google.com/file/d/1CoZp1F0FqB3pOqYlQ7X9XqCChppZFUps/view?usp=sharing)  |
| **Deduplicated dataset**  | 74.2 MB   | 58,227  | [Link](https://drive.google.com/file/d/19JXTPTqV9gaKzHqGdbyyEuEfXD2l5DCc/view?usp=sharing)  |
| **Thread urls**           | 6.4 MB    | 95,324  | [Link](https://drive.google.com/file/d/1AmVIqCnWzSCqexTv02wOBAnPhiTkgHkP/view?usp=sharing)  |
| **GPT-2 Checkpoints**     | ~1.5 GB   |   | [Link](https://drive.google.com/drive/folders/1RUV2gPbGUetBFlIJZ9_W-ARB70x_9s-L?usp=sharing)  |





:toolbox: Requirements
===
See `requirements.txt`.


:gear: Installation
===
```
$ git clone https://github.com/davidsvy/Neural-Scam-Artist
$ cd Neural-Scam-Artist
$ pip install -r requirements.txt
```

:roll_of_paper: Usage
===

To generate dataset (~3 hours on Colab):
```

$ python create_dataset.py [-c configs/create_dataset.yaml]
```

To deduplicate dataset (~30 minutes on Colab):
```
$ python deduplicate_dataset.py [-c configs/deduplicate_dataset.yaml]
```

To train GPT-2 (~3 hours/epoch on Colab with K80):
```
$ python gpt2_train.py [-c configs/gpt2_train.yaml]
```

To generate text with GPT-2:
```
$ python gpt2_sample.py [-c configs/gpt2_sample.yaml]
```

