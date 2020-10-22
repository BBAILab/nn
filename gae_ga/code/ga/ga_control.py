# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 09:31:22 2020

@author: jrbrad
"""

import subprocess
import multiprocessing
#import pathlib
import time
import argparse
import glob
import re
import os


def run_this(q, mnist_id, model_file, weights_file, out_folder, folder):
    #path = pathlib.Path(__file__).parent.absolute()
    result = subprocess.run(['python', folder + '/code/ga_mnist_adv_worker.py', str(mnist_id), model_file, weights_file, out_folder, folder], stdout=subprocess.PIPE)
    #result = subprocess.run(['python', 'C:/Users/jrbrad/Desktop/adversarial_eg/ga/cl/new/ga_mnist_adv_worker.py', str(mnist_id), model_file, weights_file, out_folder], stdout=subprocess.PIPE)
    result = result.stdout.decode('utf-8')
    q.put(result)
    return result

#def parse(x):
#    x = x.rstrip(']').lstrip('[']).split(',')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate adversarial examples for neural network')
    parser.add_argument('start_id', metavar='start_id', type=int, help='Starting MNIST index for evaluation')
    parser.add_argument('end_id', metavar='end_id', type=int, help='Ending MNIST index for evaluation')
    parser.add_argument('num_proc', metavar='num_proc', type=str, help='Number of processes')
    parser.add_argument('file_model', metavar='file_model', type=str, help='JSON file for neural network model')
    parser.add_argument('file_weights', metavar='file_weights', type=str, help='h5 file for neural network wieghts')
    parser.add_argument('folder', metavar='folder', type=str, help='base file folder for code/input/output subfolders')
    args = parser.parse_args()
    
    try:
        num_proc = int(args.num_proc) #18 #4
        print('Creating pool with %d processes\n' % num_proc)
    except:
        print('Argument for number of processes cannot be converted to an integer. \n')
    num_progs = args.end_id - args.start_id + 1
    model_file = args.file_model    #'ff_mnist.json'
    weights_file = args.file_weights  #'ff_mnist.h5'
    filename_stub = re.sub('.json','',model_file) 
        
    
    ''' Find unique file name for output and establish it '''
    try:
        subfold_out = os.environ['COMPUTERNAME'] + '/'
    except:
        subfold_out = 'hpc/'
        
    nums = re.compile('[0-9]+\.csv')
    out_folder = args.folder + 'output/' + subfold_out
    #out_folder = 'C:/Users/jrbrad/Desktop/adversarial_eg/ga/cl/output\\'
    files = glob.glob(out_folder + filename_stub + '*.csv')
    if len(files) == 0:
      ext = str(0)
    else:
        for i in range(len(files)):
            files[i] = int(nums.search(files[i]).group(0).rstrip('.csv'))
        ext = str(max(files) + 1)
    
    
    
    output_file = out_folder + filename_stub + ext + '.csv'
    f = open(output_file,'w')
    f.write('')
    f.close()
    

    with multiprocessing.Pool(num_proc) as pool:
        
        m = multiprocessing.Manager()
        q = m.Queue()
        
        TASKS = [(q, i, model_file, weights_file, out_folder, args.folder) for i in range(args.start_id, args.end_id + 1)] 
        
            
        #results1 = pool.starmap(run_this, TASKS)
        # Trial with async
        results1 = pool.starmap_async(run_this, TASKS)
        print('DONE')
        #results = [pool.apply_async(calculate, t) for t in TASKS]
        #imap_it = pool.imap(calculatestar, TASKS)
        #imap_unordered_it = pool.imap_unordered(calculatestar, TASKS)
    
        '''
        print('starmap() results:')
        for r in results1:
            try:
                print('\t try ', r.get())
            except:
                print('\t except ', r)
        print() '''
        
        '''
        for i in range(1, num_progs):
            print("result", i, ":", q.get())'''
        
        # Trial with async
        #print(results1)
        
        num_retrieve = 0
        while num_retrieve < num_progs:
            try:
                #print('checking')
                result = q.get()
                print("result", num_retrieve, ":", result)
                num_retrieve += 1
                #result = result.rstrip(']').lstrip('[').split(',')
                f = open(output_file,'a')  #, buffering=0
                f.write(result)
                f.write('\n')
                f.close()
                #result = q.get()
            except:
                time.sleep(2)
        