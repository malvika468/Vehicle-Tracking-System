# Import package
import paho.mqtt.client as mqtt
import ssl
import time
from PIL import Image
from array import array
import io

# Define Variables
MQTT_PORT = 8883
MQTT_KEEPALIVE_INTERVAL = 45
MQTT_TOPIC = "helloTopic"
MQTT_MSG = "hello MQTT"

MQTT_HOST = "a1f7n44iwz132t.iot.us-west-2.amazonaws.com"
CA_ROOT_CERT_FILE = "root-CA.crt"
THING_CERT_FILE = "certificate.pem.crt"
THING_PRIVATE_KEY = "private.pem.key"


# Define on_publish event function
def on_publish(client, userdata, mid):
	print"Message Published..."

def on_subscribe(mosq, obj, mid, granted_qos):
    print"Subscribed to Topic: "
# Initiate MQTT Client


def on_message(mosq, obj, msg):
	#print "Topic: " + str(msg.topic)
	#print ("QoS: " + str(msg.qos))
	print str(msg.payload)

mqttc = mqtt.Client()
# Register publish callback function
mqttc.on_publish = on_publish
mqttc.on_subscribe = on_subscribe
mqttc.on_message = on_message
# Configure TLS Set
mqttc.tls_set(CA_ROOT_CERT_FILE, certfile=THING_CERT_FILE, keyfile=THING_PRIVATE_KEY, cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLSv1_2, ciphers=None)

# Connect with MQTT Broker
mqttc.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
#mqttc.subscribe(MQTT_TOPIC, 0)		
mqttc.loop_start()

counter = 0
while True:
	f = open('image0.jpg',"rb")
	imagestring = bytearray(f.read())
	f.close()
	mqttc.publish(MQTT_TOPIC,imagestring,qos=1)
	counter += 1
	time.sleep(5)

# Disconnect from MQTT_Broker
# mqttc.disconnect()