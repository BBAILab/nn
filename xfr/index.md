# The Transferability of Adversarial Examples Within and Across Neural Network Architectures
## By James R. Bradley and Aaron Blossom

- Draft of the paper
- Code
  - [Training code](https://wm1693.box.com/s/gcxhl50qsj2stz1xec3uk0jesnnpwi72){:target="_blank"}
    - The link above will download a zip file of the Python code used to train 100 instantiations of each of the predicting networks used in the article along with CSV files with adversarial example data that the programs read.
  - [Adversarial examples](https://wm1693.box.com/s/sdq916ubbxcd1y69t3dp073zfb54o0q1){:target="_blank"}
    - The link above will download a zip file of the adversarial examples and true labels for each AE types included in the article, each in a numpy .npy file which can be loaded with the numpy.load() statement.
    - More conveniently, this zip file also includes a Python file named unpack_adv_eg.py, which displays the adversarial examples for all AE types for 10 randomly selected MNIST images.
  - [Adversarial Example Generation](https://github.com/BBAILab/nn/tree/main/gae_ga/code){:target="_blank"}
    - The link above is to a repository with the code used to generate the adversarial examples for both genetic algorithms (ga folder) and the Madry-PGD AE type (madry folder).