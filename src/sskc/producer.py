from time import sleep
from json import dumps
from kafka import KafkaProducer

producer = KafkaProducer(bootstrap_servers=['localhost:9092,localhost:9093,localhost:9094'],
                         value_serializer=lambda m: dumps(m).encode('utf-8'))
for e in range(1000):
    data = {'number': e}
    producer.send('numtest', value=data)
    sleep(5)
