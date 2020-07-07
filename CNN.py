# Importing section
from keras import models
from keras import layers
from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from keras import backend as K
import os
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
# End importing section

# CNN summary
conv_base = VGG16(weights='imagenet',
            include_top=False,
            input_shape=(150, 150, 3))
conv_base.summary()
# End CNN summary

# Importing images data
base_dir = 'path_to_images' # In this part you should split the directory into test, train and validation with the images 
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20
# End Importing images data

# Extracting feature based on Chollet, 2018
def extract_features(directory, sample_count):
    
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(directory,
                                            target_size=(150, 150),
                                            batch_size=batch_size,
                                            class_mode='binary')
    i = 0
    
    for inputs_batch, labels_batch in generator:
        
        print(labels_batch)
        
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        
        if i * batch_size >= sample_count:
            
            break
            
    return features, labels
    
train_features, train_labels = extract_features(train_dir, 70)
validation_features, validation_labels = extract_features(validation_dir, 40)
test_features, test_labels = extract_features(test_dir, 60)
# End Extracting feature based on Chollet, 2018

# Preparing the images
train_features = np.reshape(train_features, (70, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (40, 4 * 4 * 512))
test_features = np.reshape(test_features, (60, 4 * 4 * 512))
# End Preparing the images

# Adding aditional layers
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
# End Adding aditional layers

# New layers summary
model.summary() 
# End New layers summary

# Obtaining train and vaidation accuracy
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                        loss='binary_crossentropy',
                        metrics=['acc'])
history = model.fit(train_features, train_labels,
epochs=30,
batch_size=20,
validation_data=(validation_features, validation_labels))
# End Obtaining train and vaidation accuracy

# Experiment to obtain accuracy
acc_set = []
val_set = []
for i in np.arange(0, 30):
    
    history = model.fit(train_features, train_labels,
    epochs=30,
    batch_size=20,
    validation_data=(validation_features, validation_labels))

    acc_set.append(history.history['acc'][-1])
    val_set.append(history.history['val_acc'][-1])
obtained_acc=pd.DataFrame({'Train Accuracy': acc_set,
              'Validation Accuracy': val_set})
obtained_acc
# End Experiment to obtain accuracy

# Experiment to obtain recall, precision and F1
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                        loss='binary_crossentropy',
                        metrics=[metrics.Precision(), metrics.Recall()])
history = model.fit(train_features, train_labels,
epochs=30,
batch_size=20,
validation_data=(validation_features, validation_labels))

prec_set = []
recall_set = []
val_prec_set = []
val_recall_set = []

for i in np.arange(0, 30):
    
    history = model.fit(train_features, train_labels,
    epochs=30,
    batch_size=20,
    validation_data=(validation_features, validation_labels))
    

    prec_set.append(history.history['precision_4'][-1])
    recall_set.append(history.history['recall_3'][-1])
    val_prec_set.append(history.history['val_precision_4'][-1])
    val_recall_set.append(history.history['val_recall_3'][-1])
obtained_rec_prec=pd.DataFrame({'Precision': prec_set,
              'Recall': recall_set,
               'val_precision':val_prec_set,
               'val_recall':val_recall_set})
 obtained_rec_prec
# End Experiment to obtain recall, precision and F1
