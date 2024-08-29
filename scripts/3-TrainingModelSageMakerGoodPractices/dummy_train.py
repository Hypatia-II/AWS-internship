#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import sys
import time
import traceback
from random import randrange

import numpy as np
import torch
from torch.utils.data import DataLoader



def train(args):
    print("done")

if __name__ == "__main__":
    # Retrieve hyperparameters
    parser = argparse.ArgumentParser()

    # Parse the hyperparameters
    parser.add_argument('--config_path', type=str, required=True)

    args = parser.parse_args()
    
    print("AAAAA", os.getcwd(), "AAAAA")
    print("AAAAA", os.system('ls') ,"AAAAA")

    # Attempt to install modules
    os.system("apt-get install libsndfile1-dev")
    

    try:
        train(args)
        sys.exit(1)
    except:
        sys.exit(0)
