#!/usr/bin/env python3

import argparse
import logging
import sys
import time

import datasets
import tqdm
import torch
import transformers


parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_file", "--o"
    default=False,
    type=bool,
    # required=True,
    help="",
)

if __name__ == '__main__':

    args = parser.parse_args()
    logger = logging.getLogger("")
    logger.setLevel(logging.DEBUG)
    if args.output_file is not None:
        fh = logging.FileHandler(args.output_file)
        sh = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s], %(levelname)s %(message)s",
            datefmt="%a, %d %b %Y %H:%M:%S",
        )
        fh.setFormatter(formatter)
        sh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(sh)

    msg = "Printing packages and versions:"
    if args.output_file is not None:
        msg = "logging to file {}\nPrinting packages and versions:\n".format(args.output_file)
    else:
        msg = "Printing versions:\n"

    logging.info("Test file checking that the configuration worked\n" + "="*56 + "\n\n")
    logging.info(msg)


    for i in range(3):
        logger.info("\tPyTorch:\t\t{}".format(torch.__version__))
        logger.info("\tTransformers:\t{}".format(transformers.__version__))
        logger.info("\tDatasets:\t{}".format(datasets.__version__))
        logger.info("-"*56 + "\n")
        time.sleep(1)
