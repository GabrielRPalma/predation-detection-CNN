# This code was based on the work of Chollet, 2018
# Importing section
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import cv2
from keras.applications.vgg16 import VGG16
from keras import backend as K
import matplotlib.pyplot as plt
# End Importing section

# Preparing the images
img_path = 'path_to_your_image'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
# End Preparing the images

# Classifying the image with VGG16 with imagenet data
preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])
np.argmax(preds[0])
# End Classifying the image with VGG16 with imagenet data

# Obtaining the feature map
model = VGG16(weights='imagenet')
cheetah = model.output[:, 293] 
last_conv_layer = model.get_layer('block5_conv3')
# End Obtaining the feature map

# Section Gradient-weighted Class Activation Mapping
grads = K.gradients(african_e66lephant_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input],[pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
# End Section Gradient-weighted Class Activation Mapping

# Applying the heatmap on the analyzed image
img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap) 
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('path_to_save_the_image', superimposed_img)
# End Applying the heatmap on the analyzed image

