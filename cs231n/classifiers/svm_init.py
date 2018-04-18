import random
import numpy as np
import matplotlib.pyplot as plt

import cs231n.classifiers.linear_classifier as linear_classifier
from cs231n.classifiers.data_util import load_cifar10
x_train,y_train,x_test,y_test=load_cifar10('../datasets/cifar-10-batches-py')

num_training=49000
num_validation=1000
num_test=1000
num_dev=500

mask=range(num_training,num_training+num_validation)
x_val=x_train[mask]
y_val=y_train[mask]
mask=range(num_training)
x_train=x_train[mask]
y_train=y_train[mask]
mask=np.random.choice(num_training,num_dev,replace=False)
x_dev=x_train[mask]
y_dev=y_train[mask]
mask=range(num_test)
x_test=x_test[mask]
y_test=y_test[mask]

x_train=np.reshape(x_train,(x_train.shape[0],-1))
x_val=np.reshape(x_val,(x_val.shape[0],-1))
x_test=np.reshape(x_test,(x_test.shape[0],-1))
x_dev=np.reshape(x_dev,(x_dev.shape[0],-1))

mean_image=np.mean(x_train,axis=0)
x_train-=mean_image
x_val-=mean_image
x_test-=mean_image
x_dev-=mean_image

x_train=np.hstack([x_train,np.ones((x_train.shape[0],1))])
x_val=np.hstack([x_val,np.ones((x_val.shape[0],1))])
x_test=np.hstack([x_test,np.ones((x_test.shape[0],1))])
x_dev=np.hstack([x_dev,np.ones((x_dev.shape[0],1))])

learning_rates=[1.4e-7,1.5e-7,1.6e-7]
regularization_strengths=[ (1+i*0.1)*1e4 for i in range(-3,3)] + [(2+0.1*i)*1e4 for i in range(-3,3)]
results={}
best_val=-1
best_svm=None
for learning in learning_rates:
    for regularization in regularization_strengths:
        svm=linear_classifier.LinearSVM()
        svm.train(x_train,y_train,learning_rate=learning,reg=regularization,num_iters=2000)
        y_train_pred=svm.predict(x_train)
        train_accuracy=np.mean(y_train==y_train_pred)
        print('training accuracy: %f' % (train_accuracy))
        y_val_pred=svm.predict(x_val)
        val_accuracy=np.mean(y_val==y_val_pred)
        print('validation accuracy: %f' % (val_accuracy))
        if val_accuracy>best_val:
            best_val=val_accuracy
            best_svm=svm
        results[(learning,regularization)]=(train_accuracy,val_accuracy)

for lr , reg in sorted(results):
    train_accuracy,val_accuracy=results[(lr,reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (lr,reg,train_accuracy,val_accuracy))
    print('best validation accuracy achieved during cross-validation: %f' % best_val)


# x_scatter=[math.log10(x[0]) for x in results] #1
# y_scatter=[math.log10(x[1]) for x in results] #2
# sz=[results[x][0]*1500 for x in results]  #3
# plt.subplot(1,2,1)
# plt.scatter(x_scatter,y_scatter,sz)
# plt.xlabel('log learning rate')
# plt.ylabel('log regularization strength')
# plt.title('cifar10 training accuracy')
# sz=[results[x][1]*1500 for x in results]
# plt.subplot(1,2,2)
# plt.scatter(x_scatter,y_scatter,sz)
# plt.xlabel('log learning rate')
# plt.ylabel('log regularization strength')
# plt.title('cifar10 validation accuracy')
# plt.show()

y_test_pred=best_svm.predict(x_test)
test_accuracy=np.mean(y_test==y_test_pred)
print('linear svm on raw pixels final test set accuracy: %f'% test_accuracy)

w=best_svm.W[:-1,:] #1
w=w.reshape(32,32,3,10)
w_min,w_max=np.min(w),np.max(w)
classes= ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
for i in range(10):
    plt.subplot(2,5,i+1)
    wimg=255.0*(w[:,:,:,i].squeeze()-w_min)/(w_max-w_min)
    plt.imshow(wimg.astype('uint8'))
    plt.axis('off')
    plt.title(classes[i])

plt.show()