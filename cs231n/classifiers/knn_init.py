import numpy as np
from cs231n.classifiers.k_nearest_neighbor import KNearestNeighbor
from cs231n.classifiers.data_util import load_cifar10
import matplotlib.pyplot as plt
x_train,y_train,x_test,y_test=load_cifar10('../datasets/cifar-10-batches-py')

print('training data shape:',x_train.shape)
print('training labels shape:',y_train.shape)
print('test data shape:',x_test.shape)
print('test labels shape:',y_test.shape)

classes=['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
num_claesses=len(classes)
samples_per_class=7
for y ,cls in enumerate(classes):
    idxs=np.flatnonzero(y_train==y)
    idxs=np.random.choice(idxs,samples_per_class,replace=False)
    # for i ,idx in enumerate(idxs):
        # plt_idx=i*num_claesses+y+1
        # plt.subplot(samples_per_class,num_claesses,plt_idx)
        # plt.imshow(x_train[idx].astype('uint8'))
        # plt.axis('off')
        # if i ==0:
        #     plt.title(cls)


num_training=50000
mask=range(num_training)
x_train=x_train[mask]
y_train=y_train[mask]
num_test=10000
mask=range(num_test)
x_test=x_test[mask]
y_test=y_test[mask]

x_train=np.reshape(x_train,(x_train.shape[0],-1))
x_test=np.reshape(x_test,(x_test.shape[0],-1))
print(x_train.shape,x_test.shape)

classifier=KNearestNeighbor()
classifier.train(x_train,y_train)

ks = range(1 , 10)
pre = []
num_correct = []
accuracy = []

for k in ks:
    pr = classifier.predict(x_test , k)
    num = np.sum(pr == y_test)
    pre.append(pr)
    num_correct.append(num)
    accuracy.append(float(num) / num_test)

plt.plot(ks , accuracy)
plt.show()

