from __future__ import unicode_literals, print_function, division
import time
import math
import re
import pickle
import unicodedata
from io import open
from random import shuffle

import torch

# import torchtext
# fast_text = torchtext.vocab.FastText(language='en')

SOS_TOKEN = 0
EOS_TOKEN = 1
MAX_LENGTH = 15

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def init_cuda():
    '''Using CUDA Device'''
    assert torch.cuda.is_available()
    device = 'CUDA'
    device_idx = torch.cuda.current_device()
    device_cap = torch.cuda.get_device_capability(device_idx)
    debug('PyTorch using {} device {}:{} with Compute Capability {}.{}'
        .format(str(device).upper(), torch.cuda.get_device_name(device_idx), 
        device_idx, device_cap[0], device_cap[1]))

def debug(message):
    print('\nDEBUG: {}\n'.format(message))

class Lang:
    def __init__(self, name):
        self.name = name
        self.word_to_idx = {}
        self.word_to_count = {}
        self.idx_to_word = {SOS_TOKEN: 'SOS', EOS_TOKEN: 'EOS'}
        self.n_words = 2
        
    def add_word(self, word):
        if word not in self.word_to_idx:
            self.word_to_idx[word] = self.n_words
            self.word_to_count[word] = 1
            self.idx_to_word[self.n_words] = word
            self.n_words += 1
        else:
            self.word_to_count[word] += 1
        
    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)
            
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s) 
        if unicodedata.category(c) != 'Mn')

def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def read_langs(lang1, lang2, reverse=False):
    
    print('Reading lines...')
    
    # Read the file and split into lines
    lines = open('data/{}-{}.txt'.format(lang1, lang2), encoding='utf-8') \
            .read().strip().split('\n')
    
    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else: 
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
        
    return input_lang, output_lang, pairs

def filter_pair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
            len(p[1].split(' ')) < MAX_LENGTH and \
            p[1].startswith(eng_prefixes)

def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]

def prepare_data(lang1, lang2, reverse=False):
    
    input_lang, output_lang, pairs = read_langs(lang1, lang2, reverse)
    print('Read {} sentence pairs'.format(len(pairs)))
    
    pairs = filter_pairs(pairs)
    print('Trimmed to {} sentence pairs'.format(len(pairs)))

    shuffle(pairs)
    
    print('Counting words...')
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs[:-1000], pairs[-1000:]

def indices_from_sentence(lang, sentence):
    return [lang.word_to_idx[word] for word in sentence.split(' ')]

def tensor_from_sentence(lang, sentence):
    indices = indices_from_sentence(lang, sentence)
    indices.append(EOS_TOKEN)
    return torch.tensor(indices, dtype=torch.long, device=torch.device('cuda')).view(-1, 1)

def tensors_from_pair(pair, input_lang, output_lang):
    input_tensor = tensor_from_sentence(input_lang, pair[0])
    target_tensor = tensor_from_sentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '{}m {:.0f}s'.format(m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return 'ETA: {} (~ {})'.format(as_minutes(rs), as_minutes(s))


"""
Pickles a python object into memory
"""
def save_pickle(obj, file_name):
	with open('{}.pkl'.format(file_name), 'wb') as file:
		pickle.dump(obj, file)

"""
Loads a Pickle object from memory
"""
def load_pickle(file_name):
	with open('{}.pkl'.format(file_name), 'rb') as file:
		return pickle.load(file)