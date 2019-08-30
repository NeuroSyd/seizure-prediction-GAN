import os
from datetime import datetime

def log(text,fn='log'):
	with open(fn,'a+') as f:
		string = str(datetime.now()) + ': ' + text + '\r\n'
		f.write(string)

def log_loss(loss,fn='log_loss'):
	with open(fn,'a+') as f:
		string = loss + '\r\n'
		f.write(string)		