# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 19:31:09 2019

@author: apblossom
"""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.callbacks import ReduceLROnPlateau
import datetime
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Parameters Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
n_epochs = 50
b_s=50
learning_rate_begin=.001
learning_rate_end=.0001
momtm=0.0
nest_v=False
n_networks=101
testsize=.2
batchsize=int(testsize*train_images.shape[0])
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Start the timer for run time
start_time = datetime.datetime.now()

start_time = datetime.datetime.now()

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# load and prepare adversarial datasets    
adv_network_list= ['FF', 'FF_Aug', 'CNN', 'CNN_Aug', 'Madry','Madry_ga']
for adv_network_name in adv_network_list:
    dataset = np.loadtxt(adv_network_name + '_adv_egs.csv', skiprows=0, delimiter=',')
    dataset_images_name='adversarial_images_' + adv_network_name
    globals() [dataset_images_name] = dataset[:,1:785].astype('float32')
    globals() [dataset_images_name]=(globals() [dataset_images_name]).reshape(((globals() [dataset_images_name]).shape[0], 28, 28, 1))
    dataset_labels_name='adversarial_labels_' + adv_network_name
    globals() [dataset_labels_name] = dataset[:,0].astype('int')
    dataset_labels_name_cat='adversarial_labels_' + adv_network_name + '_cat'
    globals() [dataset_labels_name_cat]=to_categorical(globals() [dataset_labels_name])


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Pretreat Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# All images will be rescaled by 1./255
train_images=train_images.astype('float32')/255
test_images=test_images.astype('float32')/255
#create a fraction of images to distort
train_images, dtrain_images, train_labels, dtrain_labels = train_test_split(train_images, train_labels, test_size = testsize)#, random_state=random_seed)
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28*28))
dtrain_images = dtrain_images.reshape((dtrain_images.shape[0], 28 , 28, 1))
dtrain_labels = dtrain_labels.reshape((dtrain_labels.shape[0],1))

train_datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

# fit parameters from data
train_datagen.fit(dtrain_images)

for dtrain_images_batch, dtrain_labels_batch in train_datagen.flow(dtrain_images, dtrain_labels, batch_size=batchsize):
    dtrain_images= dtrain_images_batch[0:dtrain_images_batch.shape[0] , 0:28, 0:28, 0]
    train_images= train_images[0:train_images.shape[0] , 0:28, 0:28, 0]
    train_images_batch=np.append(train_images,dtrain_images) 
    train_images=train_images_batch.reshape((train_images.shape[0]+dtrain_images.shape[0]),28,28,1)
    train_images=train_images.reshape(((train_images.shape[0]),784))
    break    

# `to_categorical` converts this into a matrix with as many
# columns as there are classes. The number of rows
# stays the same.
train_labels=np.append(train_labels,dtrain_labels_batch)
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
#print(train_images.shape)

# load edge dataset
dataset = np.loadtxt("FF-Aug_adv_egs.csv", skiprows=0, delimiter=',')
#dataset = pd.read_csv('C:/Users/Aaron Paul Blossom/OneDrive/AI Robustness Research/FF__adv_egs/FF__adv_egs.csv', sep=',',header=None)
#print(dataset.shape)

adversarial_images = dataset[:,1:785]
adversarial_labels = dataset[:,0]
adversarial_images=adversarial_images.astype('float32')
adversarial_labels=adversarial_labels.astype('int')
adversarial_labels=to_categorical(adversarial_labels)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
network_name='FF_Aug'
short_file_name= network_name +' with adversarial results'
misclassified_file_name= short_file_name+ ' misclassified'
properly_classified_file_name = short_file_name + ' properly_classified'

ff= open(short_file_name+'.txt','w+')
header=('type' +','+'adv'+','+'network_number' +','+ 'test_loss' +','+  'test_accuracy' +','+ 'adversarial_loss' +','+ 'adversarial_accuracy' +"\n")
ff.write(header)

gg= open(misclassified_file_name+'.txt','w+')
gheader=('type' +','+'adv'+','+'network_number'+','+'serial_number'+','+'predicted' +','+ 'true' +"\n")
gg.write(gheader)

hh= open(properly_classified_file_name+'.txt','w+')
hheader=('type' +','+'adv'+','+'network_number'+','+'serial_number'+','+'predicted' +','+ 'true' +"\n")
hh.write(gheader)

hh.close()
gg.close()    
ff.close()

for Index in range(1,n_networks):
    opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    # Set a learning rate annealer
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                                patience=3, 
                                                verbose=1, 
                                                factor=0.5, 
                                                min_lr=0.00001)
    #Create the feedforward  netowrk
    network = models.Sequential()
    #Add the first hidden layer specifying the input shape
    network.add(layers.Dense(800, activation='relu', input_shape=(28 * 28,)))
    network.add(BatchNormalization())
    network.add(layers.Dense(400, activation='relu'))
    network.add(BatchNormalization())
    network.add(layers.Dense(10, activation='softmax'))
    network.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    #print a summary of the model
    #network.summary()
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Fit Model Section
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #Train the model
    history=network.fit(train_images, train_labels, epochs=n_epochs, batch_size=b_s, verbose=0,
              validation_data=(test_images, test_labels), callbacks=[learning_rate_reduction])

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Show output Section
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
     """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Show output Section
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    ff= open(short_file_name+'.txt','a+')
    gg= open(misclassified_file_name+'.txt','a+')
    hh= open(properly_classified_file_name+'.txt','a+')
    
    for adv_network_name in adv_network_list:
        adv_images_name='adversarial_images_' + adv_network_name
        adv_images=globals()[adv_images_name]
        adv_labels_name='adversarial_labels_' + adv_network_name
        adv_labels=globals()[adv_labels_name]
        adv_labels_name_cat='adversarial_labels_' + adv_network_name + '_cat'
        adv_labels_cat=globals()[adv_labels_name_cat]
        adversarial_loss, adversarial_accuracy = model.evaluate(adv_images, adv_labels_cat, verbose=0)
        test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
        results = (network_name + ',' + adv_network_name + ',' + str(Index) + ',' + str(test_loss) + ',' + str(test_accuracy) + ',' + str(adversarial_loss) +','+ str(adversarial_accuracy)+"\n")
        ff.write(results)
    
        #create predictions on the adversarial set
        predicted_classes =model.predict_classes(adv_images)
       
        #See which we did not predicted correctly
        correct_indices = np.nonzero(predicted_classes == adv_labels)[0]
        incorrect_indices = np.nonzero(predicted_classes != adv_labels)[0]
        
        # print to file incorrect predictions
        for i, incorrect in enumerate(incorrect_indices[:]):
            misclassified = (network_name+','+ adv_network_name +','+str(Index)+','+str(incorrect)+','+str(predicted_classes[incorrect]) +','+ str(adv_labels[incorrect]) +"\n")
            gg.write(misclassified)
        # print to file correct predictions
        for i, correct in enumerate(correct_indices[:]):
            properly_classified = (network_name+','+ adv_network_name +','+str(Index)+','+str(correct)+','+str(predicted_classes[correct]) +','+ str(adv_labels[correct]) +"\n")
            hh.write(properly_classified)
    hh.close()
    gg.close()    
    ff.close()
#calculate and print the time to run
stop_time = datetime.datetime.now()
print ("Time required for training:",stop_time - start_time)
