# Import all the necessary files!
import os
import tensorflow as tf
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras import Model  # type: ignore
from flask import Flask, request, abort
import cv2
import json
import numpy as np
from flask_cors import CORS
from tensorflow.python.keras.backend import set_session  # type: ignore

import base64
from datetime import datetime

from linebot import (
    LineBotApi, WebhookParser
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    BeaconEvent
)
from pyngrok import ngrok
import os
import sys
from dotenv import load_dotenv

import sqlite3


def CreateTable():
    ''' Create status table, with 1 row of of 'status' '''
    with sqlite3.connect('status.db') as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS status (status text)''')
        c.execute('''INSERT INTO status VALUES ('AWAY')''')
        conn.commit()


def UpdateStatus(status):
    ''' Update status table '''
    with sqlite3.connect('status.db') as conn:
        c = conn.cursor()
        c.execute('''UPDATE status SET status = ?''', (status,))
        conn.commit()


def GetStatus():
    ''' Get status from status table '''
    with sqlite3.connect('status.db') as conn:
        c = conn.cursor()
        c.execute('''SELECT status FROM status''')
        status = c.fetchone()[0]
        return status


# !Face Recognition 🌝
graph = tf.get_default_graph()

app = Flask(__name__)
CORS(app)
sess = tf.Session()
set_session(sess)

model = tf.keras.models.load_model('facenet_keras.h5')
# model.summary()


def img_to_encoding(path, model):
    img1 = cv2.imread(path, 1)
    img = img1[..., ::-1]
    dim = (160, 160)
    # resize image
    if (img.shape != (160, 160, 3)):
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    x_train = np.array([img])
    print(x_train)
    embedding = model.predict(x_train)
    # print("embedding",embedding)
    return embedding


database = {}
# database["test1"] = img_to_encoding("images/test1.jpg", model)
# database["test2"] = img_to_encoding("images/test2.jpg", model)
# database["test3"] = img_to_encoding("images/test3.jpg", model)


def verify(image_path, identity, database, model):

    encoding = img_to_encoding(image_path, model)
    dist = np.linalg.norm(encoding-database[identity])
    print(dist)
    if dist < 5:
        print("It's " + str(identity) + ", welcome in!")
        match = True
    else:
        print("It's not " + str(identity) + ", please go away")
        match = False
    return dist, match


# !LINE BOT 🤖
# get channel_secret and channel_access_token from your environment variable
load_dotenv()
channel_secret = os.getenv('LINE_CHANNEL_SECRET', None)
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN', None)

line_bot_api = LineBotApi(channel_access_token)
parser = WebhookParser(channel_secret)


@app.route("/callback", methods=['POST'])
def callback():
    with sqlite3.connect('status.db') as conn:
        c = conn.cursor()
        signature = request.headers['X-Line-Signature']

        # get request body as text
        body = request.get_data(as_text=True)
        app.logger.info("Request body: " + body)

        # parse webhook body
        try:
            events = parser.parse(body, signature)
        except InvalidSignatureError:
            abort(400)

        # if event is MessageEvent and message is TextMessage, then echo text
        for event in events:
            print(event)
            if isinstance(event, MessageEvent):
                if isinstance(event.message, TextMessage):
                    line_bot_api.reply_message(
                        event.reply_token,
                        TextSendMessage(text=event.message.text)
                    )
            if isinstance(event, BeaconEvent):
                print('Got beacon event')

                # If the user is entering the beacon range, update the status to 'HOME'
                if event.beacon.type == 'enter':
                    print('enter')
                    UpdateStatus('HOME')
                    line_bot_api.reply_message(
                        event.reply_token,
                        TextSendMessage(text='Welcome home!'))

                    # Add an interval to check if the user is still at home, every 10 seconds

                # # If the user is leaving the beacon range, update the status to 'AWAY'
                # elif event.beacon.type == 'leave':
                #     print('leave')
                #     UpdateStatus('AWAY')
                #     line_bot_api.reply_message(
                #         event.reply_token,
                #         TextSendMessage(text='Bye bye!'))

        return 'OK'


@app.route('/register', methods=['POST'])
def register():
    try:
        username = request.get_json()['username']
        img_data = request.get_json()['image64']
        with open('images/'+username+'.jpg', "wb") as fh:
            fh.write(base64.b64decode(img_data[22:]))
        path = 'images/'+username+'.jpg'

        global sess
        global graph
        with graph.as_default():
            set_session(sess)
            database[username] = img_to_encoding(path, model)
        return json.dumps({"status": 200})
    except:
        return json.dumps({"status": 500})


def who_is_it(image_path, database, model):
    print(image_path)
    encoding = img_to_encoding(image_path, model)
    min_dist = 1000
    identity = None
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        dist = np.linalg.norm(encoding-db_enc)
        print(dist)
        if dist < min_dist:
            min_dist = dist
            identity = name
    if min_dist > 5:
        print("Not in the database.")
    else:
        print("it's " + str(identity) + ", the distance is " + str(min_dist))
    return min_dist, identity


@app.route('/verify', methods=['POST'])
def change():
    if GetStatus() == 'HOME':
        img_data = request.get_json()['image64']
        img_name = str(int(datetime.timestamp(datetime.now())))
        with open('images/'+img_name+'.jpg', "wb") as fh:
            fh.write(base64.b64decode(img_data[22:]))
        path = 'images/'+img_name+'.jpg'
        global sess
        global graph
        with graph.as_default():
            set_session(sess)
            min_dist, identity = who_is_it(path, database, model)
        os.remove(path)
        if min_dist > 5:
            return json.dumps({"identity": 0})
        return json.dumps({"identity": str(identity)})
    else:
        return json.dumps({"identity": str(0)})


@app.route('/status', methods=['GET'])
def status():
    status = GetStatus()
    return json.dumps({"status": status})


CreateTable()
ngrok.set_auth_token(os.environ['NGROK_AUTH_TOKEN'])
http_tunnel = ngrok.connect(5000)
endpoint_url = http_tunnel.public_url.replace('http://', 'https://')
print('LINE bot online at ' + endpoint_url)
line_bot_api.set_webhook_endpoint(endpoint_url + '/callback')

if __name__ == '__main__':
    app.run(port=5000, debug=True, use_reloader=False)
