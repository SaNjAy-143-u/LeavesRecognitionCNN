import os
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('tf')
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

class loadDataset(object):
	def __init__(self,tp):
		pass
	@staticmethod
	def getData(inp_path,img_size,num_channel=1):
		if(inp_path==None):
			PATH = os.getcwd()
			data_path = PATH + '/data'
			data_dir_list = os.listdir(data_path)
		else:
			data_dir_list=os.listdir(inp_path)
		
		num_samples = len(data_dir_list)
		print(num_samples)
		img_data_list=[]
		for img in data_dir_list:
			input_img=cv2.imread(data_path+'/'+ img )
			input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
			input_img_resize=cv2.resize(input_img,img_size)
			img_data_list.append(input_img_resize)
		label_list = np.ones((num_samples,),dtype=int)
		label_list[0:59] = 0
		label_list[59:122] = 1
		label_list[122:194] = 2
		label_list[194:267] = 3
		label_list[267:323] = 4
		label_list[323:385] = 5
		label_list[385:437] = 6
		label_list[437:496] = 7
		label_list[496:551] = 8
		label_list[551:616] = 9
		label_list[616:666] = 10
		label_list[666:729] = 11
		label_list[729:781] = 12
		label_list[781:846] = 13
		label_list[846:906] = 14
		label_list[906:962] = 15
		label_list[962:1039] = 16
		label_list[1039:1101] = 17
		label_list[1101:1162] = 18
		label_list[1162:1228] = 19
		label_list[1228:1288] = 20
		label_list[1288:1343] = 21
		label_list[1343:1398] = 22
		label_list[1398:1463] = 23
		label_list[1463:1517] = 24
		label_list[1517:1569] = 25
		label_list[1569:1622] = 26
		label_list[1622:1677] = 27
		label_list[1677:1734] = 28
		label_list[1734:1798] = 29
		label_list[1798:1851] = 30
		label_list[1851:1907] = 31

		img_data = np.array(img_data_list)
		img_data = img_data.astype('float32')
		img_data /= 255
		Y = np_utils.to_categorical(label_list, 32)
		if num_channel==1:
			if K.image_dim_ordering()=='th':
				img_data= np.expand_dims(img_data, axis=1) 
				print (img_data.shape)
			else:
				img_data= np.expand_dims(img_data, axis=4) 
				print (img_data.shape)
				
		else:
			if K.image_dim_ordering()=='th':
				img_data=np.rollaxis(img_data,3,1)
				print (img_data.shape)

		images,labels = shuffle(img_data,Y)
		X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)
		return  X_train, X_test, y_train, y_test,32










