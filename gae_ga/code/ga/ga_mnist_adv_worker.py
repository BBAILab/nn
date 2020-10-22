# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:08:12 2020

@author: jrbrad
"""

import matplotlib.pyplot as plt
import random
#import glob

#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from keras import models
from keras.utils import to_categorical
from keras.datasets import mnist

import numpy as np
import json
import argparse
import os
import datetime
import re

def jsonKeys2int(x):
    if isinstance(x, dict):
            return {int(k):v for k,v in x.items()}
    return x

class GA():
    
    def new(self,mnist_index):
        ''' Define target '''
        self.mnist_index = mnist_index
        self.target_img = self.train_images[mnist_index]  # numpy array of length 784
        self.target_label = self.train_labels[mnist_index] # numpy array of length 10, on-hot encoded
        self.target_label_dig = np.argmax(self.target_label)
        
        ''' Create random population '''
        if self.rand_mode == 'rand':
            self.pop = np.array([self.rand_mnist_w_constraint() for i in range(self.pop_size)], dtype='float32')
        elif self.rand_mode == 'mad':
            with open(self.in_folder + 'mad_dist.json', 'r') as f:
                f_mad_dict = json.load(f, object_hook=jsonKeys2int)
            self.f_mad_list = []
            for i in range(len(f_mad_dict)):
                self.f_mad_list.append([(v,k) for k,v in f_mad_dict[i].items()])
            del f_mad_dict
            self.pop = np.array([self.rand_mnist_mad_w_constraint() for i in range(self.pop_size)], dtype='float32')
        else:
            if self.max_img:
                print('Infeasible population initializtion.  Population not created.')
        if self.fit_type == 'mad':
            #self.pop_fit = [self.fitness_mad(p) for p in self.pop]
            #self.pop_fit = 1/np.sum(np.divide(np.abs(np.subtract(self.pop, self.target_img)), self.mad[self.target_label_dig]), axis=1)
            self.pop_fit = self.fit_mad()
        elif self.fit_type == 'ssd':
            #self.pop_fit = [self.fitness(p) for p in self.pop]
            #self.pop_fit = 1/np.sum(np.power(np.subtract(self.pop, self.target_img), 2.0), axis = 1)
            self.pop_fit = self.fit_ssd()
        else:
            if self.max_img:
                print('Invalid fit_type in GA.__init__')
            exit(1)
        
        self.compute_c_o_prob()
        self.max_fit = 0
        self.get_max_fit(self.max_img)
        if self.max_img:
            print('Random population created')      
        
    
    #def __init__(self, pop_size, num_gen, prob_mut_genome, prob_mut_pixel, prob_wht, prob_blk, gen, target_index, input_folder, output_folder, loaded_model, max_img, log_name, fit_type = 'mad', min_mad = 0.1, rand_mode = 'rand'):
    def __init__(self, pop_size, num_gen, prob_mut_genome, prob_mut_pixel, prob_wht, prob_blk, gen, input_folder, model_filename, weights_filename, output_folder, max_img, fit_type = 'mad', min_mad = 0.1, rand_mode = 'rand'):
        ''' Set general parameters '''
        self.pop_size = pop_size
        self.num_gen = num_gen
        self.min_mad = min_mad
        self.fit_type = fit_type
        max_mad = 0.15
        self.rand_mode = rand_mode
        self.in_folder = input_folder
        self.out_folder = output_folder
        self.max_img = max_img
        #self.log_name = log_name
        self.rand_mode = rand_mode
        ''' re object for stripping filename characters '''
        self.re_strip = re.compile('_B\d+E\d+')
        
        ''' Load Model & Weights, and Compile '''
        json_file = open(input_folder + model_filename, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = models.model_from_json(loaded_model_json)
        self.loaded_model.load_weights(input_folder + weights_filename)
        self.loaded_model.compile(optimizer='rmsprop',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        if self.max_img:
            print("Model loaded from disk and compiled")

        
        ''' Load and pre-treat MNIST data '''
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = mnist.load_data()
        #Reshape to 60000 x 784
        self.train_images = self.train_images.reshape((60000, 28 * 28))
        self.test_images = self.test_images.reshape((10000, 28 * 28))
        
        # Normalize range of data to 0,1
        self.train_images = self.train_images.astype('float32') / 255
        self.test_images = self.test_images.astype('float32') / 255
        
        # `to_categorical` converts this into a matrix with as many
        # columns as there are classes. The number of rows
        # stays the same.
        #self.mnist_index = target_index
        self.train_labels = to_categorical(self.train_labels)
        self.test_labels = to_categorical(self.test_labels)
        #self.target_img = self.train_images[target_index]  # numpy array of length 784
        #self.target_label = self.train_labels[target_index] # numpy array of length 10, on-hot encoded
        #self.target_label_dig = np.argmax(self.target_label)
        
        if self.max_img:
            print ("train_images.shape",self.train_images.shape)
            print ("len(train_labels)",len(self.train_labels))
            print("train_labels",self.train_labels)
            print("test_images.shape", self.test_images.shape)
            print("len(test_labels)", len(self.test_labels))
            print("test_labels", self.test_labels)
    
            print('MNIST data loaded and pre-conditioned')
        
        
        ''' Create MAD data '''
        if self.fit_type == 'mad':
            try:     # Try to read the MAD data
                f = open(self.in_folder + 'mnist_mad.csv','r')
                data = f.readlines()
                f.close()
                for i in range(len(data)):
                    data[i] = data[i].strip().split(',')
                    for j in range(len(data[i])):
                        data[i][j] = float(data[i][j])
                self.mad = [np.array(data[i]).reshape(784,) for i in range(len(data))]
            except:    # If no file, create the data and save it to a file
                indices = np.arange(10, dtype=int)
                obs = [False for i in range(10)]
                for i in range(len(self.train_labels)):
                    this_dig = np.matmul(self.train_labels[i],indices).astype(int)
                    if isinstance(obs[this_dig], bool):
                        obs[this_dig] = self.train_images[i]
                    else:
                        obs[this_dig] = np.vstack([obs[this_dig], self.train_images[i]])
                        
                # This is a list of pixel medians with one element for each digit type
                self.mad = [np.median(obs[i], axis=0) for i in range(len(obs))] 
                
                # Compute absolute deviations from the pixel medians
                self.mad = [np.abs(np.subtract(obs[i], self.mad[i])) for i in range(len(obs))]
                
                # Compute medians of absolute differences/deviations, that is, MAD
                self.mad = [np.median(self.mad[i], axis=0) for i in range(len(obs))]
                
                # Set minimum value for MAD to avoid division by zero
                self.mad = np.minimum(np.maximum(self.mad, self.min_mad),max_mad)
                
                del obs
                
                f = open(self.in_folder + 'mnist_mad.csv','w')
                for i in range(len(self.mad)):
                    my_str = ''
                    for j in range(self.mad[i].shape[0]):
                        my_str += str(self.mad[i,j]) + ', '
                    my_str = my_str.rstrip(', ')
                    f.write(my_str + '\n')
                f.close()
        
        ''' Load Model and Compile '''
        #self.loaded_model = loaded_model
        '''
        json_file = open(self.in_folder + model_filename, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = models.model_from_json(loaded_model_json)
        #del loaded_model_json
        # load weights into new model
        self.loaded_model.load_weights(self.in_folder + weights_filename)
        
        #Compile
        self.loaded_model.compile(optimizer='rmsprop',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        print("Model loaded from disk and compiled")
        '''
        
        ''' Set random population parameters'''
        self.prob_mut_genome = prob_mut_genome
        self.prob_mut_pixel = prob_mut_pixel
        self.prob_wht = prob_wht
        self.prob_blk = prob_blk
        '''
        if rand_mode == 'rand':
            self.pop = np.array([self.rand_mnist_w_constraint() for i in range(pop_size)], dtype='float32')
        elif rand_mode == 'mad':
            with open(self.in_folder + 'mad_dist.json', 'r') as f:
                f_mad_dict = json.load(f, object_hook=jsonKeys2int)
            self.f_mad_list = []
            for i in range(len(f_mad_dict)):
                self.f_mad_list.append([(v,k) for k,v in f_mad_dict[i].items()])
            del f_mad_dict
            self.pop = np.array([self.rand_mnist_mad_w_constraint() for i in range(pop_size)], dtype='float32')
        else:
            print('Infeasible population initializtion.  Population not created.')
        if self.fit_type == 'mad':
            #self.pop_fit = [self.fitness_mad(p) for p in self.pop]
            #self.pop_fit = 1/np.sum(np.divide(np.abs(np.subtract(self.pop, self.target_img)), self.mad[self.target_label_dig]), axis=1)
            self.pop_fit = self.fit_mad()
        elif self.fit_type == 'ssd':
            #self.pop_fit = [self.fitness(p) for p in self.pop]
            #self.pop_fit = 1/np.sum(np.power(np.subtract(self.pop, self.target_img), 2.0), axis = 1)
            self.pop_fit = self.fit_ssd()
        else:
            print('Invalid fit_type in GA.__init__')
            exit(1)
        
        self.max_fit = 0
        self.get_max_fit(self.max_img)
        if self.max_img:
            print('Random population created')
        '''
        
        # Create output file
        # Filename is  the label_index_numgen_popsize_probWht_probBlk_probMutGenome(x100)_probMutPixel(x1000)_fileIndex
        #self.filename = str(self.target_label_dig) + '_' + str(self.num_gen) + '_' + str(self.pop_size) + '_' + str(int(self.prob_wht * 100)) + '_' + str(int(self.prob_blk * 1000)) + '_' + 'constraint' + '_' + self.fit_type
        #files = glob.glob('./output/' + self.filename  + '*')
        #self.filename += '_' + str(len(files)) #+ '.csv'
        
    def pop_condition(self):
        # condition population to  remove images classified same as benchmark
        old_changed = range(len(self.pop))
        while len(old_changed):
            new_changed = []
            for i in old_changed:
                if np.argmax(self.loaded_model.predict(np.array([self.pop[i]]))) == self.target_label_dig:
                    self.pop[i] = np.array(self.rand_mnist())
                    new_changed.append(i)
            old_changed = [x for x in new_changed]
        return

    def rand_mnist(self):
        #blkProb = self.prob_blk
        #whtProb = 0.25/(1 - self.prob_blk)
        #img =  [[ 0.0 if random.random() <= blkProb else 1.0 if random.random() <= whtProb else random.randint(1,255)/255 for j in range(28)] for i in range(28)]
        #img =  [0.0 if random.random() <= blkProb else 1.0 if random.random() <= whtProb else random.randint(1,255)/255 for i in range(784)]
        img =  [self.rand_pixel() for i in range(784)]
        return np.array(img)
    
    def rand_mnist_w_constraint(self):
        img =  [self.rand_pixel() for i in range(784)]
        while self.check_target_class(np.array(img)):
            img =  [self.rand_pixel() for i in range(784)]
        return np.array(img)
    
    def rand_mnist_mad_w_constraint(self):
        img =  [self.rand_mad_pixel(i) for i in range(784)]
        while self.check_target_class(np.array(img)):
            img =  [self.rand_mad_pixel(i) for i in range(784)]
        return np.array(img)
    
    def rand_pixel(self):
        return 0.0 if random.random() <= self.prob_blk else 1.0 if random.random() <= self.prob_wht else random.randint(1,255)/255
    
    def rand_mad_pixel(self,i):
        x = random.random()
        j = 0
        while self.f_mad_list[i][j][0] < x:
            j +=1
        return self.f_mad_list[i][j][1]/255
     
    '''
    def fitness(self,cand):
        return 1/np.sum((self.target_img - cand)**2)
    
    def fitness_mad(self, img):
        return 1/np.sum(np.divide(np.abs(np.subtract(self.target_img, img)),self.mad[self.target_label_dig]))
    '''
    
    def get_max_fit(self, show_image):
        #max_fit_gen = 0
        #fit_ind_gen = -1
        
        # self.pop_fit is a numpy array
        
        fit_ind_gen = np.argmax(self.pop_fit)
        if self.pop_fit[fit_ind_gen] > self.max_fit:
            self.max_fit = self.pop_fit[fit_ind_gen]
            self.max_fit_img = self.pop[fit_ind_gen]
            if self.max_img:
                print('New best max. fitness: ' + str(self.max_fit))
            if show_image:
                print('Target Image')
                self.show_image(self.target_img)
                print('Population Best Fit')
                print('Classification : ' + str(np.argmax(self.loaded_model.predict(np.array([self.max_fit_img])))))
                self.show_image(self.max_fit_img)
        else:
            if self.max_img:
                print('No fitness improvement.')
        
        '''
        for i in range(len(self.pop_fit)):
            if self.pop_fit[i] > max_fit_gen:
                max_fit_gen = self.pop_fit[i]
                max_fit_img = self.pop[i]
                fit_ind_gen = i
        if fit_ind_gen != -1:
            if max_fit_gen > self.max_fit:
                self.max_fit = max_fit_gen;
                self.max_fit_img = max_fit_img
                print('New best max. fitness: ' + str(self.max_fit))
                if show_image:
                    print('Target Image')
                    self.show_image(self.target_img)
                    print('Population Best Fit')
                    print('Classification : ' + str(np.argmax(self.loaded_model.predict(np.array([self.max_fit_img])))))
                    self.show_image(self.max_fit_img)
            else:
                print('No fitness improvement.')
        '''
            
        return
    
    def show_image(self,img):
        plt.imshow(img.reshape(28,28), cmap='gray')
        plt.show()
        return
    
    def pick_parents(self):
        rn = [random.random(), random.random()]
        rn.sort()
        parents = []
        i = 0
        ind = 0
        while len(parents) < 2:
            while rn[ind] > self.pop_cum_prob[i]:
                i += 1
            parents.append(i)
            ind += 1
            if (ind < 2) and (rn[ind] < self.pop_cum_prob[i]):
                parents.append(i)
        return parents
    
    def compute_c_o_prob(self):
        self.pop_cum_prob = self.pop_fit / np.sum(self.pop_fit)
        self.pop_cum_prob = np.cumsum(self.pop_cum_prob)
        self.pop_cum_prob[-1] = 1.0
        ''' Most recently changed the lines below for those above '''
        #sum_o_fit = sum(self.pop_fit)
        #pop_prob = [f/sum_o_fit for f in self.pop_fit]
        #self.pop_cum_prob = [sum(pop_prob[0:i+1]) for i in range(len(pop_prob) - 1)]
        #self.pop_cum_prob.append(1.0)
        return
    
    '''
    def next_gen(self):
        self.compute_c_o_prob()
        self.crossover()
        self.mutate()
        if self.fit_type == 'mad':
            self.pop_fit = [self.fitness_mad(p) for p in self.pop]
        else:
            self.pop_fit = [self.fitness(p) for p in self.pop]
        self.get_max_fit(self.max_img)
        return
    '''
    
    def mutate_one(self,img):
        if random.random() <= self.prob_mut_genome:
                for j in range(len(self.pop[0])):
                    if random.random() <= self.prob_mut_pixel:
                        img[j] = self.rand_pixel()
        
    def mutate_one_pop(self,pop):
        for i in range(len(pop)):
            if random.random() <= self.prob_mut_genome:
                for j in range(len(self.pop[0])):
                    if random.random() <= self.prob_mut_pixel:
                        pop[i][j] = self.rand_pixel()
        
    def mutate_one_mad(self,img):
        if random.random() <= self.prob_mut_genome:
                for j in range(len(self.pop[0])):
                    if random.random() <= self.prob_mut_pixel:
                        img[j] = self.rand_mad_pixel(j)
        
    def mutate_one_mad_pop(self,pop):
        for i in range(len(pop)):
            if random.random() <= self.prob_mut_genome:
                for j in range(len(self.pop[0])):
                    if random.random() <= self.prob_mut_pixel:
                        pop[i][j] = self.rand_mad_pixel(j)
        
    def next_gen_w_contraint(self):
        self.compute_c_o_prob()
        
        # get best of previous generation
        pop_fit_here = [(self.pop_fit[i], i) for i in range(len(self.pop_fit))]
        pop_fit_here.sort(reverse=True)
        best_dim = min(15,len(pop_fit_here))
        sort_ind = [x[1] for x in pop_fit_here[:best_dim]]
        digits = np.argmax(self.loaded_model.predict(self.pop[sort_ind]), axis=1)
        
        #digits = [(np.argmax(self.loaded_model.predict(np.array([self.pop[pop_fit_here[i][1]].reshape(784,)]))),i) for i in range(best_dim)]
        best_ind = [99 for i in range(10)]
        for i in range(len(digits)-1,-1,-1):
            best_ind[digits[i]] = pop_fit_here[i][1]
            #best_ind[digits[i][0]] = digits[i][1]
        count_good = sum([1 for x in best_ind if x <= 9])
        
        parents = [tuple(self.pick_parents()) for i in range(self.pop_size - count_good)]
        crossover = np.random.randint(0,self.pop[0].shape[0] - 1, size = (self.pop_size - count_good,))
        new_pop = [np.concatenate([self.pop[parents[i][0]][0:crossover[i]], self.pop[parents[i][1]][crossover[i]:]]) for i in range(self.pop_size - count_good)]
        if self.rand_mode == 'rand':
            self.mutate_one_pop(new_pop)
        elif self.rand_mode == 'mad':
           self.mutate_one_mad_pop(new_pop)
        
        for i in range(len(best_ind)):
            if best_ind[i] <= 9:
                new_pop.append(self.pop[best_ind[i]])

        new_pop = np.array(new_pop, dtype = 'float32')        
        target_match = self.check_target_class_pop(new_pop)
        
        for i in range(len(new_pop)):
            if target_match[i]:
                if random.random() <=0.5:
                    new_pop[i] = parents[i][0]
                else:
                    new_pop[i] = parents[i][1]
        
        ''' Old code '''
        '''
        while len(new_pop) < self.pop_size - count_good:
            p1,p2 = self.pick_parents()
            cross_pt = random.randint(0,len(self.pop) - 1)   # Possible error: should be self.pop[0]
            # crossover 
            candidate = np.concatenate([self.pop[p1][0:cross_pt], self.pop[p2][cross_pt:]])
            if self.rand_mode == 'rand':
                self.mutate_one(candidate)
            elif self.rand_mode == 'mad':
                self.mutate_one_mad(candidate)
            if self.check_target_class(candidate):
                new_pop.append(self.pop[p1])
                new_pop.append(self.pop[p2])
            else:
                new_pop.append(candidate)'''
        
        #new_pop = new_pop[:self.pop_size - 1]
        #new_pop.append(self.max_fit_img)  #put best fit image from this generation in next generation
                
        self.pop = new_pop.copy()
        
        if self.fit_type == 'mad':
            self.pop_fit = self.fit_mad() #[self.fitness_mad(p) for p in self.pop]
        else:
            self.pop_fit = self.fit_ssd() #[self.fitness(p) for p in self.pop]
        self.get_max_fit(self.max_img)
        
        return 
    
    def fit_mad(self):
        return 1/np.sum(np.divide(np.abs(np.subtract(self.pop, self.target_img)), self.mad[self.target_label_dig]), axis=1)
        
    def fit_ssd(self):
        return 1/np.sum(np.power(np.subtract(self.pop, self.target_img), 2.0), axis = 1)
    
    def evolve(self):
        start_time = datetime.datetime.now()
        for i in range(self.num_gen):
            if self.max_img:
                print('Generation ' + str(i+1) + ':  ', end='')
            self.next_gen_w_contraint()
            
            ''' Write Results to File '''
            #f_out.write(str(i) + ',' + str(self.max_fit) + ',' + str(np.mean(self.pop_fit)) + '\n')
        
        ''' Write Best Fit Results to File'''
        combine_fit_pop = [(self.pop_fit[i], self.pop[i], np.argmax(self.loaded_model.predict(np.array([self.pop[i]])))) for i in range(self.pop_size)]
        combine_fit_pop.sort(reverse=True, key = lambda x:x[0])
        
        done = False
        max_fit = combine_fit_pop[0][0]
        digit_done = [False for i in range(10)]
        best_images = np.array([combine_fit_pop[0][1]])
        digit_done[combine_fit_pop[0][2]] = True
        i = 1
        best_labels = [combine_fit_pop[0][2]]
        while not done:
            if combine_fit_pop[i][0] > 0.90 * max_fit:
                if digit_done[combine_fit_pop[i][2]] == False:
                    digit_done[combine_fit_pop[i][2]] = True
                    best_images = np.vstack([best_images,combine_fit_pop[i][1]])
                    best_labels.append(combine_fit_pop[i][2])
                i += 1
            else:
                done = True
                
        '''
        f = open('./output/images/' + self.filename + '_fit.csv','w')
        for i in range(1,15):
            f.write(str(combine_fit_pop[i][0]) + '\n')
            best_images = np.vstack([best_images,combine_fit_pop[i][1]])
        f.close()
        np.savetxt('./output/images/' + self.filename + '_images.csv', best_images, delimiter=",",fmt='%8.6f',newline='\n')
        '''
        
        finish_time = datetime.datetime.now()
        
        ''' Report Results to console '''
        if self.max_img:
            print('\n\n\n\nResults')
            print('Maximum fitness: ' + str(self.max_fit) + '\n')
            print('Target Image')
            self.show_image(self.target_img)
                
            print('Best Fit Images')
            #print('Best Fit Classification: ' + str(np.argmax(self.loaded_model.predict(np.array([self.max_fit_img])))))
            #self.show_image(self.max_fit_img)
            for i in range(len(best_images)):
                print('Image label:' + str(best_labels[i]))
                self.show_image(best_images[i])
        
        
        
        ''' Put best into database '''
        '''
        cnx = MySQL.connect(user='root', passwd='MySQL', host='127.0.0.1', db='adv_exmpl')
        cursor = cnx.cursor(buffered=True)
        for i in range(len(best_images)):
            img_str = ''
            for j in range(best_images[i].shape[0]):
                img_str += str(best_images[i][j]) + ' '
            cursor.callproc('spInsertRow',(self.mnist_index,int(self.target_label_dig),int(best_labels[i]), img_str)) # best_images[i].tobytes()
            cnx.commit() '''
            
        ''' Create output '''
        finish_time = datetime.datetime.now()
        elapse_time = finish_time - start_time
        #f_out = open(self.out_folder + self.log_name + '.csv','a')
        #log_name_stripped = self.re_strip.sub('',self.log_name)
        output = []
        for i in range(len(best_images)):
            #img_str = '"'
            img_str = ''
            for j in range(best_images[i].shape[0]):
                img_str += str(best_images[i][j]) + ' '
            output.append(str(self.mnist_index) + ', ' + str(self.target_label_dig) + ', ' + str(best_labels[i]) + ', ' + str(self.max_fit) + ',' + str(elapse_time) + ', ' + img_str + '\n')
            #img_str += '"'
            #out_str = log_name_stripped  + ', ' + str(self.mnist_index) + ', ' + str(self.target_label_dig) + ', ' + str(best_labels[i]) + ', ' + str(self.max_fit) + ', ' + img_str + '\n'
            #f_out.write(out_str)
        #f_out.close()
        
        ''' Write to log file '''
        #finish_time = datetime.datetime.now()
        #elapse_time = finish_time - start_time
        #minutes = int(elapse_time.total_seconds()/60)
        #seconds = elapse_time.total_seconds() - minutes * 60.0
        #f_out = open(self.out_folder + self.log_name + '.log','a')
        #f_out.write('MNIST index ' + str(self.mnist_index) + ' completed at ' + datetime.datetime.strftime(datetime.datetime.now(), '%m/%d/%y %H:%M:%S') + ' in ' + str(minutes) + ':' + str(seconds) + ' with fitness ' + str(self.max_fit) + '\n')
        #f_out.close()
        
        #new_log_name = re.sub('E\d+', 'E' + str(self.mnist_index), self.log_name)
        
        #os.rename(self.out_folder + self.log_name + '.csv', self.out_folder + new_log_name + '.csv')
        #os.rename(self.out_folder + self.log_name + '.log', self.out_folder + new_log_name + '.log')
        
        #self.log_name = new_log_name
        
        return output
        
    def mutate(self):
        for i in range(len(self.pop)):
            if random.random() <= self.prob_mut_genome:
                for j in range(len(self.pop[0])):
                    if random.random() <= self.prob_mut_pixel:
                        self.pop[i,j] = self.rand_pixel()
        return
    
    def crossover(self):
        keep_best_fit = True
        new_pop = []
        for i in range(len(self.pop)):
            p1, p2 = self.pick_parents()
            cross_pt = random.randint(0,len(self.pop) - 1)
            new_pop.append(np.concatenate([self.pop[p1][0:cross_pt], self.pop[p2][cross_pt:]]))
            '''
            if cross_pt not in [0,784]:
                new_pop.append(self.pop[p1][0:cross_pt] + self.pop[p2][cross_pt:])
            elif cross_pt == 0:
                new_pop.append(self.pop[p1])
            else:
                new_pop.append(self.pop[p2])'''
        self.pop = np.array([p for p in new_pop])
        
        if keep_best_fit:
            self.pop = np.vstack([self.pop[0:len(new_pop)-1], self.max_fit_img])
        return

    def check_target_class(self, img):
        return np.argmax(self.loaded_model.predict(np.array([img]), batch_size=1000)) == self.target_label_dig
    
    def check_target_class_pop(self, pop):
        return np.argmax(self.loaded_model.predict(pop, batch_size=1000), axis = 1) == self.target_label_dig
    
""" Function Definitions """

""" Create random grayscale images with pixels on [0.0, 1.0] """
def randDig():
    blkProb = 0.7
    whtProb = 0.25/(1 - blkProb)
    #img =  [[ 0.0 if random.random() <= blkProb else 1.0 if random.random() <= whtProb else random.randint(1,255)/255 for j in range(28)] for i in range(28)]
    img =  [0.0 if random.random() <= blkProb else 1.0 if random.random() <= whtProb else random.randint(1,255)/255 for i in range(784)]
    return np.array(img)

"""
def fitness(bench,cand):
    ssq = 0.0
    for i in range(len(bench)):
        ssq += (bench[i] - cand[i])^2
    return 1/ssq"""
    
def fitness(target,cand):
    return 1/np.sum((target - cand)**2)



''' Handle input arguments '''
parser = argparse.ArgumentParser(description='Generate adversarial examples for neural network')
parser.add_argument('mnist_id', metavar='mnist_id', type=int, help='MNIST index for evaluation')
#parser.add_argument('end', metavar='end', type=int, help='Ending MNIST index')
parser.add_argument('file_model', metavar='file_model', type=str, help='JSON file for neural network model')
parser.add_argument('file_weights', metavar='file_weights', type=str, help='h5 file for neural network weights')
parser.add_argument('out_folder', metavar='out_folder', type=str, help='file folder for output')
parser.add_argument('folder', metavar='folder', type=str, help='base file folder for code/input/output subfolders')
args = parser.parse_args()

''' Set GA parameters '''
pop_size = 1000   # population size
prob_mut_genome = 0.7    # probability of mutation
genome_size = 784
prob_mut_pixel = 2.0 / genome_size
num_gen = 2000       # number of generations
max_fit = 0       # maximum fitness
pop_fit = []      # population fitness
#target_index = 10
model_filename = args.file_model #'ff_mnist.json'
weights_filename = args.file_weights #'ff_mnist.h5'
prob_wht = 0.25
prob_blk = 0.7
min_mad = 0.1    # Values tried: 0.001, 0.05
fit_type = 'mad'
rand_type = 'mad'
        

'''
if os.environ['COMPUTERNAME'] == 'BRADLEYJ-5810':
    input_folder = 'D:/research/neuralnetworks/code/mnist/adversarial_eg/ga/input/'
    output_folder = 'D:/Research/NeuralNetworks/code/MNIST/adversarial_eg/ga/cl/output/'
    prints_img = True
else:
    input_folder = '/sciclone/home10/jrbrad/files/mnist/input/'
    output_folder = '/sciclone/home10/jrbrad/files/mnist/output/'
    prints_img = False
'''

input_folder = args.folder +  'input/' #'/sciclone/home10/jrbrad/files/mnist/input/'
#input_folder = 'C:/Users/jrbrad/Desktop/adversarial_eg/ga/input/'
output_folder = args.out_folder #'/sciclone/home10/jrbrad/files/mnist/output/'
prints_img = False
    
#log_name = re.sub('_','-',re.sub('\.json','',model_filename)) + '_' + re.sub('_','-',re.sub('\.h5','',weights_filename)) + '_' + 'pop' + str(pop_size) + '_' + 'mutgenome' + str(int(prob_mut_genome*100)) + '_' + 'mutpix' + str(int(prob_mut_pixel*1000)) + '_' + 'gen' + str(num_gen) + '_' + 'fit-' + fit_type + '_' + 'rand-' + rand_type + '_B' + str(args.start) + 'E' + str(args.start)
''' Create empty output file '''
#f_out = open(output_folder + log_name + '.csv','w')
#f_out.close()
''' Create empty log file '''
#f_out = open(output_folder + log_name + '.log','w')
#f_out.close()

scen_name = re.sub('_','-',re.sub('\.json','',model_filename)) + '_' + 'pop' + str(pop_size) + '_' + 'mutgenome' + str(int(prob_mut_genome*100)) + '_' + 'mutpix' + str(int(prob_mut_pixel*1000)) + '_' + 'gen' + str(num_gen) + '_' + 'fit-' + fit_type + '_' + 'rand-' + rand_type
#  + '_' + re.sub('_','-',re.sub('\.h5','',weights_filename))

''' Instantiate GA object '''
#ga = GA(pop_size, num_gen, prob_mut_genome, prob_mut_pixel, prob_wht, prob_blk, num_gen, i, input_folder, output_folder, loaded_model, prints_img, log_name, fit_type, min_mad, 'mad')
ga = GA(pop_size, num_gen, prob_mut_genome, prob_mut_pixel, prob_wht, prob_blk, num_gen, input_folder, model_filename, weights_filename, output_folder, prints_img, fit_type, min_mad, rand_type)
ga.new(args.mnist_id)
result = ga.evolve()
for i in range(len(result)):
    result[i] = scen_name + ',' + result[i]
print(result, end='')