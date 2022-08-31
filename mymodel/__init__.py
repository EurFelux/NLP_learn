import torch
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("clue/albert_chinese_tiny")

# 将BERT的[CLS]作为bos，[SEP]作为eos
tokenizer.bos_token = '[CLS]'
tokenizer.bos_token_id = 101
tokenizer.eos_token_ = '[SEP]'
tokenizer.eos_token_id = 102

MAX_LEN = 30
device = 'cuda' if torch.cuda.is_available() else 'cpu'