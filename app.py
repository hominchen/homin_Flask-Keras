#!/usr/bin/env python
# -*- coding=utf-8 -*-
from flask import Flask, request, render_template
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input, decode_predictions,VGG16
from keras.applications.resnet_v2 import ResNet50V2

app = Flask(__name__)
# model = VGG16()
model = ResNet50V2()

@app.route("/")
def hello():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "static/images/" + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    yhat = model.predict(image)
    label = decode_predictions(yhat)
    label = label[0][0]

    classification = '%s (%.2f%%)' % (label[1], label[2]*100)

    return render_template('index.html', 
        prediction=classification,
        image_path=image_path)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
