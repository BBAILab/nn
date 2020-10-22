# Madry et al. Neural Network Models and Code

This folder contains code related to the Madry et al. PGD method for generating adversarial examples.  It contains code from 
the MadryLab GitHub repositoty, [GitHub_MadryLab](https://github.com/MadryLab/mnist_challenge), that we have revised and also 
code that we have written based on the work of Madry et al.


The code has been presented in multiple folders, each one containing a self-contained module of code for a particular purpose:

- <p font-family:"Courier New">madry_adv_eg</p>: contains our adaptaion of Madry wt al.'s code that we used for
  - Training the Madry neural network
  - Generating adversarial examples
- <p font-family:"Courier New">madry_validate</p>: contains code for generating PGD adversarial examples and computing the percentage of adversarial examples that the originating network properly classifies