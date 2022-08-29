#!/usr/bin/env python
# -*- coding=utf-8 -*-
from flask import Flask, request, render_template
# import numpy as np
# import tensorflow as tf

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input, decode_predictions,VGG16


app = Flask(__name__)
model = VGG16()

# model = tf.compat.v1.keras.applications.vgg16.VGG16()
# model.summary()

@app.route("/")
def hello():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)
    
    # 我的tensorflow
    # img = tf.keras.preprocessing.image.load_img('image_path', 
    # target_size=(224, 224))

    # # 轉成numpy
    # image = tf.keras.preprocessing.image.img_to_array(img)
    # # 轉維度
    # image = image.reshape((1, image.shape[0], 
    #     image.shape[1], 
    #     image.shape[2]))
    # # 圖送入模型
    # image = tf.compat.v1.keras.applications.vgg16.preprocess_input(image)
    # # 預測
    # predict2 = model.predict(image)
    # predict = np.argmax(predict2[0])
    # return render_template('index.html', predict)

    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    yhat = model.predict(image)
    label = decode_predictions(yhat)
    label = label[0][0]

    classification = '%s (%.2f%%)' % (label[1], label[2]*100)

    return render_template('index.html', 
        prediction=classification)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
