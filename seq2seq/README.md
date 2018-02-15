NMT seq2seq model
==========================

This project is an implementation of a vanilla NMT model in MxNet. It is based on:

- a stacked LSTM encoder
- a stacked LSTM decoder

This version doesn't have attention yet.
 
The reference model configuration is:

- 2 LSTM encoder layers
- 2 LSTM decoder layers
- hidden state of 512 units
- embedding size of 512 units
- truncating the vocabulary to the top 50,000 words
- truncating the maximum sentence length to 50 words

The reference model is normally trained on the WMT15 English-German dataset, which consists of:

- raw Europarl v7
- Common Crawl
- News Commentary v7

----------

How to run the scripts?
-------------------------------

 1. Run the `get_nmt_data.sh` script to download and preprocess the dataset. Preprocessing includes corpora concatenation, shuffling, and a trainining/validation set split.
 2. Run the preprocessing step. Fair warning: this can take close to half an hour, but it's a one-time cost. Run `python preprocess_data.py` to do that. Make sure you have enough RAM, at least 32 GB.
 3. Run the model. Note that the batch size setting is global, not per GPU. So, when running on a DGX-1 with a batch size of 128 per GPU, choose a batch size of 128 * 8 = 1,024. Here is a sample command to run the training:

```python seq2seq_bucketing.py --num-layers 2 --num-embed 512 --num-hidden 512 --optimizer adam --disp-batches 1 --gpus 0,1 --num-epochs 1 --batch-size 256```
