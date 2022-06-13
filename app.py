import os
import base64
import cv2
import json
import binascii
import numpy as np
import flask
from flask import Flask, request
from tensorflow.keras.models import model_from_json
from tensorflow.keras import Model
import recognize


def base64_to_cv2image(image_string, image_extension):
    blob = base64.decodebytes(image_string.encode('ascii'))
    nparr = np.frombuffer(blob, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def cv2image_to_base64(img, img_extension):
    data = cv2.imencode(img_extension, img)[1].tobytes()
    data = base64.encodebytes(data).decode('ascii')
    return data

def recognize_logs(source_img):
    '''
    source_img - cv2 image in RGB format
    '''
    size = (256, 256)
    #img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
    #default_size = source_img.shape
    img = cv2.resize(source_img, size, interpolation=cv2.INTER_AREA)
    img = np.expand_dims(img, axis=0)

    predict = model.predict(img, batch_size=1)
    img = predict[0,:,:,0] * 255
    return img.astype(np.uint8)
    #cv2.imwrite(f'{app_path}{sep}model{sep}test.png', img)




app = Flask(__name__)
sep = os.path.sep
app_path = os.getcwd()
model_path = f'{app_path}{sep}model'

model = None
with open(f'{app_path}{sep}model{sep}model.json', 'r') as file:
    model = model_from_json(file.read())
    model.load_weights(f'{app_path}{sep}model{sep}model.h5')

#new_im_path = f'{app_path}{sep}model{sep}'
#new_im_name = 'im_34.jpg'
#size = (256, 256)
#img = cv2.imread(new_im_path+new_im_name)

@app.route('/')
@app.route('/index')
def index():
    #img_url = flask.url_for('static', filename='/root/server/model/test.png')
    return f'Hello!'

@app.route('/get_img')
def get_img():
    img = cv2.imread(f'{model_path}{sep}test.png')
    blob = cv2image_to_base64(img, '.png')
    return json.dumps({"image": blob, 'extension': '.png'})

@app.route('/sendImage', methods=['POST'])
def send_img():
    #print(request.data)
    data = json.loads(request.data) 
    img = base64_to_cv2image(data["image"], data["extension"])
    def_size = img.shape
    mask = recognize_logs(img)
    img = cv2.resize(img, (256, 256))
    ext = '.png'
    res = recognize.get_mask(img, mask)
    blob = cv2image_to_base64(res['img_with_mask'], ext)
    res = recognize.count_value(mask, res['stack_mask'], def_size)

    return json.dumps({
    	'image': blob, 
	'extension': ext,
	'value': str(res['value']),
	'log_count': str(res['log_count']),
	'coeff': str(res['coeff'])
	})



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
        #try:
        #    app.run(debug=True, host='0.0.0.0')
        #except BaseException:
        #    print('Hello')


