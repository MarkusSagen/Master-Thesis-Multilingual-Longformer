#!/usr/bin/env python

from tqdm import tqdm
import time

if __name__ == '__main__':
    print('Training a very advanced model...')
    for x in tqdm(range(1000)):
        time.sleep(1)
        print(x)

    print('Done!')
