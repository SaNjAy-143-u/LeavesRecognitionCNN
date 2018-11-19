from keras import backend as K
K.set_image_dim_ordering('tf')
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import BatchNormalization 
from keras.optimizers import SGD,RMSprop,adam

class model(object):
	def __init__(self,num_classes,num_layers,input_shape,momentum,lossfn,decay,lr):	
		self.model=Sequential();
		self.model.add(Convolution2D(32, (5,5),border_mode='same',input_shape=input_shape))
		self.model.add(PReLU(weights=None, alpha_initializer="zero"))
		self.model.add(MaxPooling2D(pool_size=(2,2)))
		self.model.add(BatchNormalization(axis=-1,momentum=momentum))
		self.model.add(Convolution2D(32, (4,4),strides=(2,2)))
		self.model.add(PReLU(weights=None, alpha_initializer="zero"))
		self.model.add(MaxPooling2D(pool_size=(2,2)))
		self.model.add(BatchNormalization(axis=-1,momentum=momentum))
		self.model.add(Convolution2D(32, (3,3), strides=(2,2)))
		self.model.add(Activation('sigmoid'))
		self.model.add(MaxPooling2D(pool_size=(2, 2),strides=(1,1)))
		self.model.add(Dropout(0.6))
		self.model.add(Flatten())
		self.model.add(Dense(num_classes))
		self.model.add(Activation('softmax'))
		sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
		self.model.compile(loss=lossfn, optimizer=sgd,metrics=["accuracy"])

	
	def forward(self,X_train,y_train,X_test,y_test,batch_size,epoch,verbose=1,plt=False,loadModel=False,saveModel=True,outDir="./"):
		if loadModel:
			self.model=load_model(outDir+'model.hdf5')
		else:
			self.hist = self.model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epoch, verbose=verbose, validation_split=0.2)
			if (plt):
				train_loss=self.hist.history['loss']
				val_loss=self.hist.history['val_loss']
				train_acc=self.hist.history['acc']
				val_acc=self.hist.history['val_acc']
				xc=range(epoch)

				plt.figure(1,figsize=(7,5))
				plt.plot(xc,train_loss)
				plt.plot(xc,val_loss)
				plt.xlabel('num of Epochs')
				plt.ylabel('loss')
				plt.title('train_loss vs val_loss')
				plt.grid(True)
				plt.legend(['train','val'])
				#print plt.style.available # use bmh, classic,ggplot for big pictures
				plt.style.use(['classic'])

				plt.figure(2,figsize=(7,5))
				plt.plot(xc,train_acc)
				plt.plot(xc,val_acc)
				plt.xlabel('num of Epochs')
				plt.ylabel('accuracy')
				plt.title('train_acc vs val_acc')
				plt.grid(True)
				plt.legend(['train','val'],loc=4)
				#print plt.style.available # use bmh, classic,ggplot for big pictures
				plt.style.use(['classic'])

		self.score = self.model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
		print('Test Loss:', self.score[0])
		print('Test accuracy:', self.score[1])


		if save_model:
			model.save('model.hdf5')
		


