import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(18,GPIO.OUT)

print("LED ON")

GPIO.output(18,GPIO.HIGH)
time.sleep(5)

print("LED OFF")

GPIO.output(18,GPIO.LOW)
