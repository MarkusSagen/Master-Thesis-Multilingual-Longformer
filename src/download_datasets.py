#!/usr/bin/env python3


import os
from os import path
import sys
import wget
from zipfile import ZipFile
import logging
import math
import random
import time


import argparse



def download_and_unzip(url, DATA_DIR=None):
    zipfile = url.rsplit('/', 1)[1].rsplit('.')[0]
    if DATA_DIR is not None:
        zipfile = f'{DATA_DIR}{zipfile}'

    if (os.path.exists(zipfile) or os.path.exists("/workspace/data/wikitext-103-raw")):
        return
    filename = wget.download(url)
    datafolder = zipfile.rsplit('.')[0]
    if path.exists(datafolder):
        return
    with ZipFile(filename, 'r') as zipObj:
       # Extract all the contents of zip file in current directory
       zipObj.extractall()


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Download zipfiles')
    parser.add_argument('--url', type=str,
                    default='https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip',
                    help='Specify the URL of the zip to download')

    parser.add_argument('--dest', type=str,
                    default='/workspace/data/',
                    help='sum the integers (default: find the max)')

    parser.add_argument('--type', type=str,
                    default='zip',
                    help='Specify type of datafile to download if known (default: zip)')

    # TODO Add support for args
    # TODO Remove hard coded values
    # TODO Add support for different types
    # TODO Test run from terminal

    args = parser.parse_args()
    #print(args.accumulate(args.integers))

    DATA_DIR = "/workspace/data/"

    # Download and unzip
    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"
    download_and_unzip(url, DATA_DIR)
