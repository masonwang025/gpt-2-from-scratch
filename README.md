# gpt-2-from-scratch

Implementing [andrej's gpt-2](https://www.youtube.com/watch?v=l8pRSuU81PU) from scratch, training using FineWeb to train. For fun, I scrape all of Eliezer's writing on LessWrong and finetune the model on it.

## _setup_

```bash
pip install -r requirements.txt
```

You can then play around with `play.ipynb` to see HF's GPT-2 architecture and to play with the shakespeare dataset.

Download data for training:

```bash
python data/fineweb.py # training data
python data/hellaswag.py # eval
python data/eliezer.py # for finetuning, scrapes all posts and comments from him
```

Recommended to use 8xA100s:

```bash
# DDP launch for e.g. 8 GPUs:
torchrun --standalone --nproc_per_node=8 train.py
# otherwise, just use python train.py

# finetune
torchrun --standalone --nproc_per_node=8 finetune.py
```
