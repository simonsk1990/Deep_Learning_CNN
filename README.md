# Deep_Learning_CNN

Simple Network for keras datasets- fashion_mnist   


60000 train images 10000 test images


It has 10 categories and we built a simple network with few layers
and checking for only small fetures:


Convelutional layer
filters=32,kernel_size=(3,3),input_shape=(28,28,1),activation=LeakyReLU(alpha=0.1)

Pooling layer
pool_size=(2,2),strides=2

Flatten layer


Dense layer
output=128 neurons,activation='relu'

Dense classes output layer
output 10 classes,activation='softmax'

    Layer (type)                 Output Shape              Param #   
    conv2d (Conv2D)              (None, 26, 26, 32)        320       
    max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
    flatten (Flatten)            (None, 5408)              0         
    dense (Dense)                (None, 128)               692352    
    dense_1 (Dense)              (None, 10)                1290     

    Total params: 693,962
    Trainable params: 693,962
    Non-trainable params: 0

on the test pics we received the folloiwng scores:

              precision    recall  f1-score   support

           0       0.82      0.91      0.86      1000
           1       0.98      0.98      0.98      1000
           2       0.83      0.86      0.84      1000
           3       0.87      0.92      0.90      1000
           4       0.86      0.82      0.84      1000
           5       0.97      0.98      0.98      1000
           6       0.78      0.66      0.72      1000
           7       0.93      0.97      0.95      1000
           8       0.98      0.98      0.98      1000
           9       0.99      0.93      0.96      1000
    accuracy                           0.90     10000          
    macro avg       0.90      0.90      0.90     10000
    weighted avg       0.90      0.90      0.90     10000 

More info in .py code
