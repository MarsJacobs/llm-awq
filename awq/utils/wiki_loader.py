import numpy as np
import torch
import os
from tqdm import tqdm
import torch.nn as nn
import math

def get_wikitext2(nsamples, seed, seqlen, model, train, enc):
    from datasets import load_dataset
    
    if train:
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    else:
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    tokenizer = enc
    
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        
    return testenc
        

def get_ptb(nsamples, seed, seqlen, model, train):
    from datasets import load_dataset
    
    if train:
        traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    else:
        valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')

    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    
    if train:
        trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
        
        import random
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
            
        return trainloader
    
    else:
        testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')
        
        return testenc


def get_c4(nsamples, seed, seqlen, model, train):
    from datasets import load_dataset
    if train:
        traindata = load_dataset(
            'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
        )
    else:
        valdata = load_dataset(
            'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
        )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    if train:
        import random
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] > seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        
        return trainloader
    
    else:
        import random
        random.seed(0)
        valenc = []
        for _ in range(256):
            while True:
                i = random.randint(0, len(valdata) - 1)
                tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
                if tmp.input_ids.shape[1] > seqlen:
                    break
            i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            valenc.append(tmp.input_ids[:, i:j])
        valenc = torch.hstack(valenc)
        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids
        valenc = TokenizerWrapper(valenc)

        return valenc 

def get_ptb_new(nsamples, seed, seqlen, model, train):
    from datasets import load_dataset
    if train:
        traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    else:
        testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    
    if train:
        trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')

        import random
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
            
        return trainloader
    else:
        testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')
        
        return testenc

def get_c4_new(nsamples, seed, seqlen, model, train):
    from datasets import load_dataset
    if train:
        traindata = load_dataset(
            'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
        )
    else:
        valdata = load_dataset(
            'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
        )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    if train:
        import random
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] > seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        
        return trainloader
    
    else:
        valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        valenc = valenc.input_ids[:, :(256 * seqlen)]

        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids
        valenc = TokenizerWrapper(valenc)

        return valenc


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model='', train=True, enc=''
):
    if 'wikitext' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, train, enc)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model, train)
        return get_ptb(nsamples, seed, seqlen, model, train)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model, train)
        return get_c4(nsamples, seed, seqlen, model, train)

@torch.no_grad()
def llama_eval(model, testenc, dev):
    print('Evaluating ...')

    losses = []
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen    
    
    for i in tqdm(range(nsamples)):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        outputs =  model(input_ids = batch, labels = batch)
        losses.append(outputs["loss"].item())
    
    eval_loss = np.mean(losses)
    ppl = math.exp(eval_loss)

    return ppl