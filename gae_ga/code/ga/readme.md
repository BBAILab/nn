# Genetic Algorithm Code

This folder contains one version of the genetic algorithm code used for generating adversarial examples.  Slight differences were required for the Feed Forward (FF) neural networks versus 
the Convolutional Neural Networks (CNN) to accommodate different input shapes.  We present just one of these versions of the code, for the FF network.  The other version for CNN is almost 
entirely the same.

The code was written as a command line program and, while developed on a Windows computer, it was deployed on a Unix-based high performance computing cluster at William & Mary.  A sample of 
a command line statement to execute the code is also given.  The code architecture used a "manager" program to manage many parallel "worker" progams, each one of which created an adversarial 
example for one MNIST image.

  - Manager program: ga_control.py
  - Worker program: ga_mnist_adv_worker.py
  - Input files:
    - FF.json
	- FF.h5
	- mad_dist.json
	- mnist_mad.csv
	
Executing ga_control.py requires these packages in an Anaconda/Python environment:

- keras
- Tensorflow 1.14 (CPU version)

The format for the command line statement is as shown below for a Windows 10 operating system.  This assumes that the command is executed from a location that will 
execute Python with the required packages as noted above avaialble.

python ga_control.py *start_index* *end_index* *num_cores* FF.json FF.h5 *folder_spec*

where

- *start_index*: index for first MNIST character to generate adversarial example
- *end_index*: index for last MNIST character to generate adversarial example 
- *num_cores*: number of cores to use
- FF.json, FF.h5: FF model data
- *folder_spec*: folder in which to looks for subfolder named *input* which contains input files and subfolder *output* where output will be written