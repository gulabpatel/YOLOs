import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


password = "briw xvqp sqgi lvue"
from_email = "gulabpatel923@gmail.com"  # must match the email used to generate the password
to_email = "gulabpatel923923@gmail.com"  # receiver email

##Server creation and authentication
server = smtplib.SMTP('smtp.gmail.com: 587')
server.starttls()
server.login(from_email, password)

## Email Send Function