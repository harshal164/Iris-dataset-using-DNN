from __future__ import print_function

import numpy as np
import tflearn
import tensorflow as tf
import matplotlib.pyplot as plt


from tflearn.data_utils import load_csv
data, labels = load_csv('iris_data.csv', target_column=4,categorical_labels=True, n_classes=3)



# Build neural network
net = tflearn.input_data(shape=[None, 4])
net = tflearn.fully_connected(net, 64)
net = tflearn.fully_connected(net, 64)
net = tflearn.fully_connected(net, 3, activation='softmax')
net = tflearn.regression(net)


train_data,test_data=tf.split(data,[130,20],0)
train_label,test_label=tf.split(labels,[130,20],0)
sess=tf.Session()
tr=sess.run(train_data)
te=sess.run(test_data)

# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(tr, labels, n_epoch=10, batch_size=8, show_metric=True,shuffle=True)

# Let's create some data for DiCaprio and Winslet
#a = [5.1,3.5,1.4,.2]
#b= [6.8,2.8,4.8,1.4]
#arr=[a,b]
count=0;


def sel_color(list):
    maximum=max(list);
    if(list[0]==maximum):
        return 'red';
    if (list[1] == maximum):
        return 'blue';
    if (list[2] == maximum):
        return 'green';

pred = model.predict(te)
print(pred)

for i in range(len(pred)):
    c1=sel_color(pred[i]);
    c2=sel_color(sess.run(test_label)[i]);
    plt.scatter(te[i][0],te[i][1] , c=c1,marker='<')
    plt.scatter(te[i][0],te[i][1] , c=c2,marker='>')
    if(c1==c2):
        count=count+1;
Accuracy=float((count*100)/(len(pred)));
print(Accuracy)
plt.title(Accuracy)
plt.show()

