import os
from io import BytesIO
# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect, send_file
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil
import torch
from lib_attr.network import ResNet50
from ReadAttributeList import ReadAttributeList
from Caption import caption_image_beam_search
from scipy.misc import imread, imresize
from visualize_single_image import vis_image
import time
import csv
import cv2
import json

# Declare a flask app
app = Flask(__name__)


# You can use pretrained model from Keras
# Check https://keras.io/applications/
# or https://www.tensorflow.org/api_docs/python/tf/keras/applications

# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
# model = MobileNetV2(weights='imagenet')
#



# # Model saved with Keras model.save()
MODEL_PATH = 'models/'
#
# # Load your own trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

'''
Load attribute model and inference
'''
num_classes_list = [ 7, 5, 6, 6, 7, 3, 3, 3, 5, 5, 5, 3, 8, 2, 2, 3, 3, 3, 3, 2, 8, 9, 4, 5, 4, 2, 3]
AttributeFile='new_edition_utf8.txt'
AttributeName, AttributeDict=ReadAttributeList(AttributeFile)
target_w = 256
target_h = 256
model=[]
for i in range(len(num_classes_list)):
    featurenum = i
    num_classes = num_classes_list[i]
    net = ResNet50(target_w, target_h, num_classes)
    net.load_state_dict(torch.load(MODEL_PATH + 'net_f3_2nl_GAP_' + str(featurenum) + '.pth'))
    model.append(net)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Image caption model and inference
# Load model
model_caption = '../a-pytorch-train/checkpoint/BEST-GYT-1209-CROP-alpha-checkpoint_thyroid_10_cap_per_img_0_min_word_freq.pth.tar'
checkpoint = torch.load(model_caption, map_location=str(device))
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()
encoder_crop = checkpoint['encoder_crop']
encoder_crop = encoder_crop.to(device)
encoder_crop.eval()
encoder_text = checkpoint['text_encoder']
# encoder_text = encoder_text.to(device)
encoder_text.eval()
word_map = 'models/WORDMAP_thyroid_10_cap_per_img_0_min_word_freq.json'
feature_map_file = 'models/FEATURE_thyroid_10_cap_per_img_0_min_word_freq.json'

# # Load word map (word2ix)
with open(word_map, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)
# Read feature map
with open(feature_map_file, 'r') as j:
    feature_map = json.load(j)

def inference(img,net):
    #data: read a image;
    # Read images

    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    assert img.shape == (3, 256, 256)
    img_tensor = torch.FloatTensor(img / 255.)
    img = torch.unsqueeze(img_tensor,0)

    features, predict = net(img)
    predict = predict.argmax(dim=1)
    return int(predict)

def attr_predict(img,model):
    pred=[]
    feature_str=''
    for i in range(len(model)):
        attr_predict = inference(img, model[i])
        feature_str = feature_str+str(attr_predict)
        pred.append(attr_predict)
    PredictResults = ''
    for i in range(27):
        Attribute_name=AttributeName[i]
        PredictResults=PredictResults+Attribute_name+':'
        PredictResults=PredictResults+AttributeDict[Attribute_name][pred[i]-1]+';'+'\n'
    print(PredictResults)
    return PredictResults, pred

def print_results(rev_word_map, seq, smooth=True):
    """
    Print caption with weights at every word.

    :param img: img sequence after reading
    :param seq: caption
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    seq = seq[0]
    words = [rev_word_map[ind] for ind in seq]
    return ''.join(words)

print('Model loaded. Check http://127.0.0.1:5000/')
img_path = "./uploads/image.png"
# Start return data to html
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        cv2.imwrite(img_path,img)
        print('Uploaded image has been saved!')

        # Make prediction
        # preds = model_predict(img, model)

        # Make attribute predict(gyt)
        preds,feat = attr_predict(img,model)
        print('Attribute classificaiton finished!!')

        # Make fine-grained predict
        fg_pred,bbox_list = vis_image()
        print('Results images have been saved!')

        feature = torch.LongTensor(feat)

        # Encode, decode with attention and beam search
        seq = caption_image_beam_search(rev_word_map, encoder, encoder_crop, encoder_text,
                                        decoder, feature, word_map, feature_map, 5)

        caption_preds = print_results(rev_word_map, seq, smooth=True)

        # preds = 'Attributes classification:' + preds + '\n' + 'Fine-grained classification:' + fg_pred
        preds = 'Attributes classification:' + preds + '\n'+'Fine-grained classification:' + fg_pred +'\n'+'Caption:'+caption_preds+'\n'
        # Serialize the result, you can add additional fields
        return jsonify(result=preds, probability=0.6)

    return None

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        # Get the image from post request
        return jsonify(result='static/detections.png')

@app.route('/crop', methods=['GET', 'POST'])
def crop():
    if request.method == 'POST':
        # Get the image from post request
        return jsonify(result='static/crop_img.png')


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # # # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()


    # # # Make fine-grained predict
    # img_path = './vis_img'
    # class_file = 'class.txt'
    # model_path = 'model_final.pt'
    # detect_image(img_path, model_path, class_file)
    # vis_image()