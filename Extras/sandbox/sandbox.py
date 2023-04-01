import numpy as np
from keras.preprocessing import image
from keras.applications import resnet50

# Load a pre-trained ResNet-50 model
model = resnet50.ResNet50(weights='imagenet')

# Load an image and preprocess it for the model
img_path = 'media/suomi-pexels-dog.jpeg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = resnet50.preprocess_input(x)

# Use the model to predict the class of the image
predictions = model.predict(x)
predicted_class = resnet50.decode_predictions(predictions, top=1)[0][0]

print('Predicted class:', predicted_class[1], ', probability:', predicted_class[2])