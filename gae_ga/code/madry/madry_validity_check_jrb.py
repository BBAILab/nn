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

if __name__ == '__main__':
  import json
  import sys
  import math
  from model import Model

  from keras.datasets import mnist

  with open('config.json') as config_file:
    config = json.load(config_file)

  model_file = tf.train.latest_checkpoint(config['model_dir'])
  print('Checkpoint loaded: %s from %s' % (model_file, config['model_dir']))
  if model_file is None:
    print('No model found')
    sys.exit()

  (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
  train_images = train_images.reshape(60000,784)
  train_images = train_images.astype('float32') / 255
  num_eval_examples =  train_images.shape[0]
      
      
  ''' read Madry adversarial examples '''
  print('Reading Madry adversarial examples')
  #madry_adv_eg_labels = np.load('pred.npy')
  madry_adv_eg_imgs = np.load('attacks/nat_trained/attack_train.npy')
      
  tf.compat.v1.reset_default_graph()
  model = Model()
  saver = tf.compat.v1.train.Saver()
  
  with tf.compat.v1.Session() as sess:
      ''' Restore the latest checkpoint
          As noted in this post,
              https://stackoverflow.com/questions/48787714/tensorflow-saver-restore-not-restoring
          models need to be restored twice for the weights stored in checkpoint files to be 
          loaded.  Otherwise, the weights are random initialized values which do not give the
          intended predictions.
      '''
      
      saver.restore(sess, model_file)
      model = Model()
      saver = tf.compat.v1.train.Saver()
      saver.restore(sess, model_file)
      
      # Iterate over the samples batch-by-batch
      eval_batch_size = config['eval_batch_size']
      num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
      
      y_pred = []
      y_mnist = []
      total_corr_pred = 0
      total_corr_mnist = 0
      
      print('Iterating over {} batches'.format(num_batches))
      
      for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)
        #print('batch size: {}'.format(bend - bstart))
        print('.', end='')
      
        x_batch = train_images[bstart:bend, :].astype(np.float32)
        y_batch = train_labels[bstart:bend].astype(np.int64)
        x_adv = madry_adv_eg_imgs[bstart:bend, :]
        
        
        dict_adv = {model.x_input: x_adv,
                    model.y_input: y_batch}
        cur_corr, y_pred_batch = sess.run([model.num_correct, model.y_pred],
                                          feed_dict=dict_adv)
      
        total_corr_pred += cur_corr
        y_pred.append(y_pred_batch)      
        
        dict_nat = {model.x_input: x_batch,
                    model.y_input: y_batch}
        cur_corr, y_pred_batch = sess.run([model.num_correct, model.y_pred],
                                          feed_dict=dict_nat)
      
        total_corr_mnist += cur_corr
        y_mnist.append(y_pred_batch)
      
y_pred = np.concatenate(y_pred, axis=0)
y_mnist = np.concatenate(y_mnist, axis=0)

print('\nPercentage adversaries identified by the originating Madry network: ' + '{:4.2f}'.format(100. * sum(y_pred == train_labels)/len(train_labels)) + '%')
print('Percentage natural MNIST images identified by the Madry network: ' + '{:4.2f}'.format(100. * sum(y_mnist == train_labels)/len(train_labels)) + '%')