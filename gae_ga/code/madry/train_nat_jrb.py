"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

''' Edited by JRB to train with only the "natural"  MNIST images rhater than
    using adversarial training '''

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from model import Model
#from pgd_attack import LinfPGDAttack

with open('config.json') as config_file:
    config = json.load(config_file)

# Setting up training parameters
tf.set_random_seed(config['random_seed'])

max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
''' JRB created the next two varialbes '''
num_adv_steps = 1000
num_acc_steps = 5
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']

batch_size = config['training_batch_size']

# Setting up the data and the model
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model()

''' JRB wrote the next block of definition statements '''
#x_attack = np.load('attack.npy') # JRB modified to not report accuracy on a previous adversarial example attack
y_mnist_test = mnist.test.labels
x_mnist_test = mnist.test.images
acc_test = []
acc_train = []
acc_attack = []
#adv_dict = {model.x_input: x_attack,
#            model.y_input: y_mnist_test}
mnist_test_dict = {model.x_input: x_mnist_test,
                   model.y_input: y_mnist_test}
mnist_train_dict = {model.x_input: mnist.train.images,
                    model.y_input: mnist.train.labels}
  
  

# Setting up the optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(model.xent,
                                                   global_step=global_step)

# Set up adversary
#attack = LinfPGDAttack(model, 
#                       config['epsilon'],
#                       config['k'],
#                       config['a'],
#                       config['random_start'],
#                       config['loss_func'])

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver(max_to_keep=3)
tf.summary.scalar('accuracy adv train', model.accuracy)
tf.summary.scalar('accuracy adv', model.accuracy)
tf.summary.scalar('xent adv train', model.xent / batch_size)
tf.summary.scalar('xent adv', model.xent / batch_size)
tf.summary.image('images adv train', model.x_image)
merged_summaries = tf.summary.merge_all()

shutil.copy('config.json', model_dir)

with tf.Session() as sess:
  # Initialize the summary writer, global variables, and our time counter.
  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
  sess.run(tf.global_variables_initializer())
  training_time = 0.0

  # Main training loop
  for ii in range(max_num_training_steps):
    x_batch, y_batch = mnist.train.next_batch(batch_size)

    # Compute Adversarial Perturbations
    #start = timer()
    #x_batch_adv = attack.perturb(x_batch, y_batch, sess)
    #end = timer()
    #training_time += end - start

    nat_dict = {model.x_input: x_batch,
                model.y_input: y_batch}

    #adv_dict = {model.x_input: x_batch_adv,
    #            model.y_input: y_batch}

    # Output to stdout
    if ii % num_output_steps == 0:
      nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
      #adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
      print('Step {}:    ({})'.format(ii, datetime.now()))
      print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
      #print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
      if ii != 0:
        print('    {} examples per second'.format(
            num_output_steps * batch_size / training_time))
        training_time = 0.0
    # Tensorboard summaries
    #if ii % num_summary_steps == 0:
      #summary = sess.run(merged_summaries, feed_dict=adv_dict)
      #summary_writer.add_summary(summary, global_step.eval(sess))

    # Write a checkpoint
    if ii % num_checkpoint_steps == 0:
      saver.save(sess,
                 os.path.join(model_dir, 'checkpoint'),
                 global_step=global_step)
      
    ''' JRB wrote the following if block '''
    if ii % num_acc_steps == 0:
        #acc_attack.append(sess.run(model.accuracy, feed_dict={model.x_input: x_attack,
        #                                                      model.y_input: y_mnist_test}))
        acc_train.append(sess.run(model.accuracy, feed_dict={model.x_input: mnist.train.images[:10000],
                                                             model.y_input: mnist.train.labels[:10000]}))
        acc_test.append(sess.run(model.accuracy, feed_dict={model.x_input: x_mnist_test,
                                                            model.y_input: y_mnist_test})) 

    # Actual training step
    start = timer()
    sess.run(train_step, feed_dict=nat_dict)   #feed_dict=adv_dict
    end = timer()
    training_time += end - start
    
    #if ii % num_adv_steps == 0:
    #  adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
    #  print('    Adversarial example accuracy {:.4}%'.format(adv_acc * 100))
    
  ''' JRB wrote all the code below this comment '''
  mnist_train_acc = sess.run(model.accuracy, feed_dict={model.x_input: mnist.train.images[:10000],
                                                        model.y_input: mnist.train.labels[:10000]})
  print('MNIST training data example accuracy {:.4}%'.format(mnist_train_acc * 100))
  
  mnist_test_acc = sess.run(model.accuracy, feed_dict=mnist_test_dict)
  print('MNIST test accuracy {:.4}%'.format(mnist_test_acc * 100))
  
  #adv_dict = {model.x_input: x_attack,
  #            model.y_input: y_mnist_test}
  #adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
  #print('Adversarial example accuracy {:.4}%'.format(adv_acc * 100))