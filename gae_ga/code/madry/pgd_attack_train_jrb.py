"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class LinfPGDAttack:
  def __init__(self, model, epsilon, k, a, random_start, loss_func):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = epsilon
    self.k = k
    self.a = a
    self.rand = random_start

    if loss_func == 'xent':
      loss = model.xent
    elif loss_func == 'cw':
      label_mask = tf.one_hot(model.y_input,
                              10,
                              on_value=1.0,
                              off_value=0.0,
                              dtype=tf.float32)
      correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
      wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax
                                  - 1e4*label_mask, axis=1)
      loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      loss = model.xent

    self.grad = tf.gradients(loss, model.x_input)[0]

  def perturb(self, x_nat, y, sess):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    if self.rand:
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
      x = np.clip(x, 0, 1) # ensure valid pixel range
    else:
      x = np.copy(x_nat)

    for i in range(self.k):
      grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                            self.model.y_input: y})

      x += self.a * np.sign(grad)

      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon) 
      x = np.clip(x, 0, 1) # ensure valid pixel range

    return x


if __name__ == '__main__':
  import json
  import sys
  import math
  import time

  ''' replacement '''
  #from tensorflow.examples.tutorials.mnist import input_data
  from keras.datasets import mnist

  from model import Model
  
  start_time = time.time()

  with open('config.json') as config_file:
    config = json.load(config_file)

  model_file = tf.train.latest_checkpoint(config['model_dir'])
  if model_file is None:
    print('No model found')
    sys.exit()

  #tf.compat.v1.reset_default_graph()  # sometimes needed for problems with loading checkpoint
  model = Model()
  attack = LinfPGDAttack(model,
                         config['epsilon'],
                         config['k'],
                         config['a'],
                         config['random_start'],
                         config['loss_func'])
  saver = tf.train.Saver()

  ''' replacement '''
  #mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
  (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
  train_images = train_images.reshape(60000,784)
  train_images = train_images.astype('float32') / 255

  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, model_file)
    #model = Model()  # this and the next two statements are sometimes needed to resolve unnsuccessful restoration of checkpoint
    #saver = tf.compat.v1.train.Saver()
    #saver.restore(sess, model_file)

    # Iterate over the samples batch-by-batch
    num_eval_examples = config['num_eval_examples_train']  # changed from 'num_eval_examples
    eval_batch_size = config['eval_batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    x_adv = [] # adv accumulator

    print('Iterating over {} batches'.format(num_batches))

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)
      print('batch size: {}'.format(bend - bstart))

      ''' replacement '''
      #x_batch = mnist.test.images[bstart:bend, :]
      #y_batch = mnist.test.labels[bstart:bend]
      x_batch = train_images[bstart:bend, :]
      y_batch = train_labels[bstart:bend]

      x_batch_adv = attack.perturb(x_batch, y_batch, sess)

      x_adv.append(x_batch_adv)

    print('Storing examples')
    path = config['store_adv_path']
    x_adv = np.concatenate(x_adv, axis=0)
    np.save(path, x_adv)
    print('Examples stored in {}'.format(path))

end_time = time.time()
print('Execution time: %s seconds.' % str(end_time - start_time))
f = open('exec_time_pgd_attacj_train_jrb.txt','w')
f.write(str(end_time - start_time) + ' seconds\n')
f.close()

f = open('attacks/nat_trained/madry_adv_eg_nat_trained.csv','w')
for i in range(x_adv.shape[0]-1):
    f.write(str(train_labels[i])+','+','.join([str(y) for y in x_adv[i].tolist()])+'\n')
f.write(str(train_labels[x_adv.shape[0]-1])+','+','.join([str(y) for y in x_adv[x_adv.shape[0]-1].tolist()]))
f.close()
    
