# gpt-2-from-scratch

Implementing GPT-2 (124M) from scratch, following [Andrej's video](https://www.youtube.com/watch?v=l8pRSuU81PU) + training using FineWeb dataset.

For fun, I scrape all of Eliezer's writing on LessWrong and finetune the model on it.

## _setup_

```bash
pip install -r requirements.txt
```

You can then play around with `play.ipynb` to see HF's GPT-2 architecture and to play with the Shakespeare dataset.

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

Run the finetuned model:

```bash
# because all it does it yaps
python yap.py
```

### _results_

**After `train.py`, this GPT model scores 30.46% on hellaswag (Eleuther Harness reports 31.14% for GPT-2 124M)** See `log/log.txt` for training logs.

Running `yap.py` will allow you to have "EliezerGPT" complete your sentences. It's honestly just nonsense, but it was fun to quickly throw this together.

```
Start the sentence: My thoughts on AGI are


Eliezer: My thoughts on AGI are that you could be on the safe side or they may lead you to the wrong area. So you may want to look into what this is and what you can do to avoid it.
- How can you tell if something is unsafe? Your answer can not be in terms of the danger it’s generating!
- How can you tell if something is an issue? When it comes to hazards and safety, there is no simple way to know what has the ability to lead to problems.
The danger of a workplace disaster is something which the worker is constantly putting on their life by doing wrong things, like not wearing their safety gear or not performing necessary equipment to help or support them.
We are going to answer the question as soon as possible. We will also discuss various types of hazards.
- Safety Hazards
- How to prevent these hazards
- Dangers in the workplace
- Safety for workers
- How to prevent falls
```
