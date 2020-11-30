import argparse
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

def parse_args(args=None):
    pass

def override_config(args):
    '''
    Override model and data configuration
    '''
    pass

def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    pass

def read_triple(file_path, entity2id, relation2id):
    triples = []
    with open(file_path) as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], entity2id[r], entity2id[t]))
    return triples
