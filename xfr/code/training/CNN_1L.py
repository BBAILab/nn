"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.utils.np_utils import to_categorical 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
import gc
import datetime
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Parameters Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

epochs = 30
n_networks = 101
batch_size = 50
testsize = 0.1
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
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
# Normalize the data
train_images = train_images / 255.0
test_images = test_images / 255.0
# Reshape image in 3 dimensions (height = 28px, width = 28px , channels = 1)
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
# Encode labels to one hot vectors
train_labels= to_categorical(train_labels)
test_labels= to_categorical(test_labels)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
network_name='CNN one layer'
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
intp_time = datetime.datetime.now()
model = Sequential()

for Index in range( 1, n_networks):
    stop_time = datetime.datetime.now()
    del model
    gc.collect()
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph() # TF graph isn't same as Keras graph
    print('Network Number = ', Index -1,"Time required for training:",datetime.datetime.now() - intp_time)
    intp_time = datetime.datetime.now()
    model = Sequential()
    model.add(Conv2D(filters = 32, activation = 'relu', kernel_size = (5,5),padding = 'Same',  input_shape = (28,28,1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    #model.add(Dropout(0.25))
    model.add(Conv2D(filters = 64, activation = 'relu', kernel_size = (3,3),padding = 'Same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    #model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dense(100, activation = 'relu'))
    #model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(10, activation = 'softmax'))
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
    #model.summary()
    # Set a learning rate annealer
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                                patience=3, 
                                                verbose=0, 
                                                factor=0.5, 
                                                min_lr=0.00001)
    
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Train Model Section
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # Fit the model
    history = model.fit(train_images,train_labels,
                                  epochs = epochs, validation_data = (test_images, test_labels),
                                  verbose = 0, callbacks=[learning_rate_reduction])
   
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




