from tensorflow.keras.datasets import fashion_mnist
import tensorflow.keras.utils as kerasUtils
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential
import numpy
from sklearn.metrics import classification_report


(X_train,y_train),(X_test,y_test) = fashion_mnist.load_data() #loading data

#Normlize data for 0-1 values and 4D:
X_train = X_train/255
X_test = X_test/255
X_train = numpy.reshape(X_train,(60000,28,28,1))
X_test = numpy.reshape(X_test,(10000,28,28,1))


#hot encoude the y to categories 0-9
#we have multiclass classification
y_categorical_test=kerasUtils.to_categorical(y_test,10)
y_categorical_train=kerasUtils.to_categorical(y_train,10)

#Building simpe model:
model = Sequential()
#most simple model, just for trainning...  1 Convolutional layer for small features


#Convelutional
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(28,28,1),activation=LeakyReLU(alpha=0.1)))
#LeakyRelu on this conv in order to remove neuron leak
#params = output_channels * (input_channels * window_size + 1) = (1*9+1)*32 = 320
#(None, 26, 26, 32) - outputsize

#Pooling
model.add(MaxPool2D(pool_size=(2,2),strides=2))
#output size = (None, 13, 13, 32)

#Flatten
model.add(Flatten())

#Dense use 2^n
model.add(Dense(128,activation='relu'))
#will use on this one the relu

#last Dense - 10 calsses with softmax for probability
model.add(Dense(10,activation='softmax'))

print(model.summary())

#Done, lets compile the model
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
#we have multiclass classification, will use cost as categorical crossentropy and the optimizer as rmsprop

model.fit(X_train,y_categorical_train,epochs=2)
#wont use steps per epoch, just run 2 epochs
print(model.evaluate(X_test,y_categorical_test))




#saving and loading model
from tensorflow.keras.models import load_model
from  sklearn.metrics import confusion_matrix, classification_report

model.save('fashion_mnist_basic_model')
myModel = load_model('fashion_mnist_basic_model')

predictions = myModel.predict_classes(X_test)



print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))



print('end')