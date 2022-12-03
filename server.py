# Import all the necessary files!
import os
import time
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
    LineBotApi, WebhookParser, WebhookHandler
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

from flask_mqtt import Mqtt, MQTT_LOG_ERR, MQTT_LOG_DEBUG


def CreateTable():
    ''' Create beacon table, with 1 row of id 0 and 'status' '''
    with sqlite3.connect('home.db') as conn:
        c = conn.cursor()
        c.execute(
            '''CREATE TABLE IF NOT EXISTS beacon (id INTEGER PRIMARY KEY, status TEXT)''')

        # check if there is already a row of id 1
        c.execute('''SELECT * FROM beacon WHERE id = 1''')
        if c.fetchone() is None:
            # if there is no row of id 1, insert a new row
            c.execute('''INSERT INTO beacon (id, status) VALUES (1, 'AWAY')''')
        else:
            # if there is a row of id 1, update the status
            c.execute('''UPDATE beacon SET status = 'AWAY' WHERE id = 1''')

        conn.commit()

    ''' Create door table, with 1 row of id 0 and 'status' '''
    with sqlite3.connect('home.db') as conn:
        c = conn.cursor()
        c.execute(
            '''CREATE TABLE IF NOT EXISTS door (id INTEGER PRIMARY KEY, status TEXT)''')

        # check if there is already a row of id 1
        c.execute('''SELECT * FROM door WHERE id = 1''')
        if c.fetchone() is None:
            # if there is no row of id 1, insert a new row
            c.execute('''INSERT INTO door (id, tatus) VALUES (1, 'CLOSE')''')
        else:
            # if there is a row of id 1, update the status
            c.execute('''UPDATE door SET status = 'CLOSE' WHERE id = 1''')

        conn.commit()

    ''' Create timestamp table, with 1 row of id 0 and 'date' '''
    with sqlite3.connect('home.db') as conn:
        c = conn.cursor()
        c.execute(
            '''CREATE TABLE IF NOT EXISTS timestamp (id INTEGER PRIMARY KEY, date TEXT )''')

        # check if there is already a row of id 1
        c.execute('''SELECT * FROM timestamp WHERE id = 1''')
        if c.fetchone() is None:
            # if there is no row of id 1, insert a new row
            c.execute('''INSERT INTO timestamp (id, date) VALUES (1, 0)''')
        else:
            # if there is a row of id 1, update the status
            c.execute('''UPDATE timestamp SET date = 0 WHERE id = 1''')

        conn.commit()


def UpdateStatus(table, status):
    ''' Update status table '''
    with sqlite3.connect('home.db') as conn:
        c = conn.cursor()
        c.execute(f'''UPDATE '{table}' SET status = '{status}' WHERE id = 1''')


def UpdateTimestamp():
    ''' Update timestamp table '''
    with sqlite3.connect('home.db') as conn:
        c = conn.cursor()
        c.execute(
            f'''UPDATE timestamp SET date = '{datetime.now()}' WHERE id = 1''')


def GetStatus(table):
    ''' Get status from status table '''
    with sqlite3.connect('home.db') as conn:
        c = conn.cursor()
        c.execute(f'''SELECT status FROM '{table}' WHERE id = 1''')
        status = c.fetchone()
        conn.commit()
    return status[0]


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


MQTT_BROKER = 'broker.hivemq.com'
app.config['MQTT_BROKER_URL'] = MQTT_BROKER  # use the free broker from NETPIE
app.config['MQTT_BROKER_PORT'] = 1883  # default port for non-tls connection
# set the time interval for sending a ping to the broker to 5 seconds
app.config['MQTT_KEEPALIVE'] = 60
# set TLS to disabled for testing purposes
app.config['MQTT_TLS_ENABLED'] = False
mqtt = Mqtt(app)
mqtt.client._client_id = mqtt.client_id.encode('utf-8')

handler = WebhookHandler(channel_secret)

mqtt_msg = ''


@app.route("/callback", methods=['POST'])
def callback():
    with sqlite3.connect('home.db') as conn:
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
                    UpdateStatus('beacon', 'HOME')
                    UpdateTimestamp()
                    line_bot_api.reply_message(
                        event.reply_token,
                        TextSendMessage(text='Welcome home!'))

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
    if GetStatus('beacon') == 'HOME':
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
    table = request.args.get('table', default='', type=str)
    if table == 'beacon':
        return json.dumps({"status": GetStatus('beacon')})
    if table == 'door':
        return json.dumps({"status": GetStatus('door')})
    return json.dumps({"status": 500})


@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    text = event.message.text
    # get user id
    user_id = event.source.user_id
    profile = line_bot_api.get_profile(user_id)
    name = profile.display_name

    if text.startswith('#'):
        if text == '#on':
            mqtt.publish('/ict792/message', '#on')
            # db = firestore.client()
            # db.collection(u'messages').add(
            #     {u'timstamp': firestore.SERVER_TIMESTAMP, u'message': name + u'turn on the LED'})

        elif text == '#off':
            mqtt.publish('/ict792/message', '#off')
            # db = firestore.client()
            # db.collection(u'messages').add(
            #     {u'timstamp': firestore.SERVER_TIMESTAMP, u'message': name + u'turn off the LED'})

        if text == '1':
            mqtt.publish('/ict792/message', '1')
        elif text == '2':
            mqtt.publish('/ict792/message', '2')


@mqtt.on_connect()
def handle_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        print('Connected successfully')
        mqtt.subscribe('/ict792/message')
    else:
        print('Bad connection. Code:', rc)


@mqtt.on_message()
def handle_mqtt_message(client, userdata, message):
    data = dict(
        topic=message.topic,
        payload=message.payload.decode()
    )
    print(
        'Received message on topic: {topic} with payload: {payload}'.format(**data))

    # # If press the board's button, send a message to LINE and save the message to firestore
    # if (data["payload"] == 'this is 💣Boom pressing!'):
    #     db = firestore.client()
    #     db.collection(u'messages').add(
    #         {u'timstamp': firestore.SERVER_TIMESTAMP, u'message': data["payload"]})


CreateTable()
ngrok.set_auth_token(os.environ['NGROK_AUTH_TOKEN'])
http_tunnel = ngrok.connect(5000)
endpoint_url = http_tunnel.public_url.replace('http://', 'https://')
print('LINE bot online at ' + endpoint_url)
line_bot_api.set_webhook_endpoint(endpoint_url + '/callback')

if __name__ == '__main__':
    app.run(port=5000, debug=True, use_reloader=False)
