import numpy as np
import json
import imageio
from os import listdir
import random

class Batcher:
    def __init__(self, path_to_data, shuffle=True, test=False):
        self.path = path_to_data
        self.classes = ['Cat', 'Dog']
        self.i = 0 #number of loaded images in the current epoch
        self.epoch = 0 #number of iteration through the dataset
        self.shuffle = shuffle
        self.tst = test
		
        self.create_file_list()
        return

    def create_file_list(self):

		#creates a list images to optimize the learning
		#shuffeled list if shuffle = True (for train and validation)

        list = []
        if not(self.tst):
            for cl in self.classes:
                list_cl = listdir(self.path + cl + '/')
                list += [self.path + cl + '/' + el for el in list_cl]
        else:
            list_cl = listdir(self.path)
            list += [self.path + '/' + el for el in list_cl]
        if self.shuffle:
            random.shuffle(list)
        self.imlist = list
        return


    def next_batch(self, batch_size=32):

		#creates a new batch of images to load for training/validation

        input = [] #contrain all the input images
        target = [] #contain all the associated labels

		#you continue loading some images until the number of images loaded get to 32 (batch_size) or until there is no image left to load
        while batch_size > 0 and self.i != len(self.imlist):
            batch_size -= 1

			#each image loaded is a numpy array of pixels of shape (W,H,C)
			#the label is in the name of the image, we use the index of the label in classes ('Cat'= 0, 'Dog' = 1)
            im = imageio.imread(self.imlist[self.i])
            label = self.classes.index(self.imlist[self.i].split('.')[2])
            self.i+=1

			#if the image has only one channel (gray scale), you must create 2 new channels with the same values
			#you must first expand the dimension of 2D to 3D
            if im.ndim == 2:
                im = np.repeat(np.expand_dims(im, axis=2), 3, axis=2)


            target.append(label)

			#entries must have the shape (N,C,H,W) for conv2d
			#N (the batch_size) dimension is created while appending each image inputs to the batch
            input.append(np.transpose(im, (2, 0, 1)))

        #all images have been loaded ---> new epoch, new shuffeled list
		#return a tuple of (batch_input, batch_targets)
        if self.i == len(self.imlist):
            self.epoch += 1
            self.i = 0
            self.create_file_list()
        return np.array(input), np.array(target)


    def next_test_batch(self, batch_size=32):

		#creates a new batch of images to load for testing
		#no targets in the testing batches

        input = []
        while batch_size > 0 and self.i != len(self.imlist):
            batch_size -= 1
            im = imageio.imread(self.imlist[self.i])
            self.i+=1
            if im.ndim == 2:
                im = np.repeat(np.expand_dims(im, axis=2), 3, axis=2)
            input.append(np.transpose(im, (2, 0, 1)))
        if self.i == len(self.imlist):
            self.epoch += 1
            self.i = 0
        return np.array(input)


    def eval(self, eval_fct, use_cuda=False):

		#first creates a new batch of images to load for validation wih next_batch
		#then evaluates the permorfance of the current model with the mean of loss and accuracy (% of well classified images) of the validation set
		#useful for the choose of hyperparameters and for checking overfitting (when validation accuracy decreases or loss increase)

        self.i = 0
        sum_loss = 0
        sum_acc = 0
        nb_loss = 0

		#only one epoch of validation after an epoch of training
        self.epoch = 0
        while self.epoch == 0:
            batch = self.next_batch(32)
            loss, acc = eval_fct(batch, use_cuda=use_cuda)
            sum_loss += loss
            sum_acc += acc
            nb_loss += 1
        return sum_loss/nb_loss, sum_acc/nb_loss


    def test(self, test_fct, use_cuda=False):

		#first creates a new batch of images to load for testing wih next_batch_test
		#returns a tuple ([images],[labels]) where labels[i] is the predicted class for images[i]

        self.epoch = 0
        self.i = 0
        out_list = []
        while self.epoch == 0:
            batch = self.next_test_batch(32)
            out = test_fct(batch, use_cuda=use_cuda)
            out_list = out_list + out #<---
        return self.imlist, out_list
