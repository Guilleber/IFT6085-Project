import argparse
import os
import datetime
import json
import math
import numpy as np
import torch as t
from torch.optim import SGD
import batcher
import model as mdl
import parameters
import matplotlib.pyplot as plt

def main(args, params, valid_acc_thresh=0):
    train_batch_loader = batcher.Batcher("./Data/train/")
    valid_batch_loader = batcher.Batcher("./Data/valid/")

    model = mdl.ConvNet(params)
    if args.use_cuda:
        model = model.cuda()

    learning_rate = params["learning_rate"]
    optimizer = SGD(model.parameters(), learning_rate)

    train_step = model.trainer(optimizer)
    valid_step = model.validator()

    tracking_valid_loss = []
    tracking_valid_acc = []
    tracking_train_loss = []
    tracking_train_loss_epoch = []
    tracking_train_acc = []
    tracking_train_acc_epoch = []
    current_epoch = 0

    while current_epoch < args.epochs:
        iteration = 0
        while current_epoch == train_batch_loader.epoch:
            batch = train_batch_loader.next_batch(batch_size=args.batch_size)

            #we print the result each 50 exemple
            if iteration%50 == 0:
                loss, acc = train_step(batch, use_cuda=args.use_cuda)
                tracking_train_loss.append(loss)
                tracking_train_acc.append(acc)
                print("Epoch: " + str(current_epoch + 1) + ", It: " + str(iteration + 1) + ", Loss: " + str(loss))
            else:
                loss, acc = train_step(batch, use_cuda=args.use_cuda)
                tracking_train_loss.append(loss)
                tracking_train_acc.append(acc)
            iteration += 1
        current_epoch += 1
        loss_valid, acc_valid = valid_batch_loader.eval(valid_step, use_cuda=args.use_cuda)
        tracking_valid_loss.append(loss_valid)
        tracking_valid_acc.append(acc_valid)
        tracking_train_loss_epoch.append(sum(tracking_train_loss)/float(len(tracking_train_loss)))
        tracking_train_loss = []
        tracking_train_acc_epoch.append(sum(tracking_train_acc)/float(len(tracking_train_acc)))
        tracking_train_acc = []
        print('\n')
        print("***VALIDATION***")
        print("Epoch: " + str(current_epoch) + ", Loss: " + str(loss_valid) + ", Acc: " + str(acc_valid))
        print("****************")
        print('\n')
        if tracking_valid_acc[-1] < valid_acc_thresh:
            break
        if current_epoch >= 3:
            """if current_epoch >= 8:
                learning_rate = learning_rate/2
                optimizer = SGD(model.parameters(), learning_rate)
                train_step = model.trainer(optimizer)
            else:"""
            if tracking_valid_loss[-2] <= tracking_valid_loss[-1]:
                learning_rate = learning_rate/2
                optimizer = SGD(model.parameters(), learning_rate)
                train_step = model.trainer(optimizer)
                print("learning rate adapted to " + str(learning_rate))
    t.save(model.state_dict(), "./models/" + args.model_name + "_acc" + str(tracking_valid_acc[-1]) + "_e" + str(current_epoch) + ".model")
    plt.plot(range(len(tracking_train_loss_epoch)), tracking_train_loss_epoch, label="train")
    plt.plot(range(len(tracking_train_loss_epoch)), tracking_valid_loss, label="valid")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
    plt.plot(range(len(tracking_train_loss_epoch)), tracking_train_acc_epoch, label="train")
    plt.plot(range(len(tracking_train_loss_epoch)), tracking_valid_acc, label="valid")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.show()
    with open("./rescc" + ("_m" if args.modified_loss else "") + ".txt", 'w') as f:
        f.write(str(tracking_train_loss_epoch))
        f.write('\n')
        f.write(str(tracking_train_acc_epoch))
        f.write('\n')
        f.write(str(tracking_valid_loss))
        f.write('\n')
        f.write(str(tracking_valid_acc))
        f.close()
    
    return tracking_valid_loss[-1], tracking_valid_acc[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IFT6135')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--use-cuda', type=bool, default=False)
    parser.add_argument('--model-name', default="model")
    parser.add_argument('--modified_loss', type=bool, default=False)

    args = parser.parse_args()
    params = parameters.params
    params["modified_loss"] = args.modified_loss
    _, _ = main(args, params)
