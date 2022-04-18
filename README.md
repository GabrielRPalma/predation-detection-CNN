<h1 align="center">Hi <img src="https://raw.githubusercontent.com/MartinHeinz/MartinHeinz/master/wave.gif" width="30px">, Intrested in automating the detection of the ecological interactions using Convolutional Neural Networks?</h1>

Here, We are going to present the R scripts and data set that can be used to reproduce the results presented in the paper "Detecting predation interaction using pretrained CNNs".

##

<h3 align="center">Programming language</h3>

<div style="display: inline_block"><br>
<img align="center" alt="Gabriel-Python" height="30" width="40" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg">
</div>

##

<h3 align="center">Examples</h3>
To run our code, first open the project "SupplementMaterial.Rproj". Then, follow the examples above to reproduce our results.


```python
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
```

##

<h3 align="center">For more details</h3>
See, <a href="https://research.thea.ie/handle/20.500.12065/3429">IMVIP 2020 : Irish Machine Vision and Image Processing Conference</a> 

