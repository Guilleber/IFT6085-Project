import argparse
import os
import datetime
import json
import math
import numpy as np
import torch as t
import batcher
import model as mdl
import parameters
import csv

def main():
    parser = argparse.ArgumentParser(description='IFT6135')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--use-cuda', type=bool, default=False)
    parser.add_argument('--model', default=None)

    args = parser.parse_args()
    params = parameters.params

    test_batch_loader = batcher.Batcher("./Data/test/", shuffle=False, test=True)

    model = mdl.ConvNet(params)
    model.load_state_dict(t.load(args.model))
    if args.use_cuda:
        model = model.cuda()

    test_step = model.tester()
    img, pred = test_batch_loader.test(test_step, use_cuda=args.use_cuda)

    with open('./Results/current_result.csv', mode='w') as csv_file:
        csv_file.write("id,label\n")
        for i in range(len(img)):
            csv_file.write(img[i].split('/')[-1][:-4] + ',' + test_batch_loader.classes[pred[i]] + '\n')
        csv_file.close()

    return


if __name__ == "__main__":
    main()
