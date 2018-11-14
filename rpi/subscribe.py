# Import package
import paho.mqtt.client as mqtt
import ssl

from PIL import Image
from array import array
import io
import mysql.connector
from datetime import datetime
import testcar
from shutil import copyfile
from PIL import Image


# Define Variables
MQTT_PORT = 8883
MQTT_KEEPALIVE_INTERVAL = 45
MQTT_TOPIC = "helloTopic"
MQTT_MSG = "hello MQTT"

MQTT_HOST = "a1f7n44iwz132t.iot.us-west-2.amazonaws.com"
CA_ROOT_CERT_FILE = "root-CA.crt"
THING_CERT_FILE = "certificate.pem.crt"
THING_PRIVATE_KEY = "private.pem.key"


def insertdb(path,l):
    db = mysql.connector.connect(user='malvika468', password='root@123',
                              host='localhost',
                              database='IOT')

    sql = 'INSERT INTO data VALUES(\''+path+'\',now(), '+str(l)+')'    
    #args = (blob_value, )
    cursor=db.cursor()
    cursor.execute(sql)
    #sql1='select * from img'
    db.commit()
    db.close()

# Define on_publish event function
def on_publish(client, userdata, mid):
	print "Message Published..."

def on_subscribe(mosq, obj, mid, granted_qos):
    print"Subscribed to Topic: "
# Initiate MQTT Client

time=datetime.now()
def on_message(mosq, obj, msg):
    global counter
    image = Image.open(io.BytesIO(msg.payload))
    image.save('image'+str(counter)+'.jpg')
    path='image'+str(counter)+'.jpg'
    #num=1
    #path='image21.jpg'
    num=testcar.predict(path)
    insertdb(str(path),num)
    num+=1
    copyfile(path,'C:/Users/user/Desktop/new_workspace/project/src/main/resources/static/img/'
                 +str(path))
    print"Image saved"
    counter = counter + 1
    
    
mqttc = mqtt.Client('malo')
# Register publish callback function
mqttc.on_subscribe = on_subscribe
mqttc.on_message = on_message
# Configure TLS Set
mqttc.tls_set(CA_ROOT_CERT_FILE, certfile=THING_CERT_FILE, keyfile=THING_PRIVATE_KEY, cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLSv1_2, ciphers=None)

# Connect with MQTT Broker
counter = 0
mqttc.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
mqttc.subscribe(MQTT_TOPIC, 0)		
mqttc.loop_forever()


# Disconnect from MQTT_Broker
# mqttc.disconnect()