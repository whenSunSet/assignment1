import numpy as np
import matplotlib.pyplot as plt
import cs231n.classifiers.linear_classifier as linear_classifier

from cs231n.classifiers.data_util import init_train_data
x_train , y_train , x_val , y_val , x_test , y_test , x_dev , y_dev = init_train_data()

learning_rates=[1.4e-7,1.5e-7,1.6e-7]
regularization_strengths=[ (1+i*0.1)*1e4 for i in range(-3,3)] + [(2+0.1*i)*1e4 for i in range(-3,3)]
results={}
best_val=-1
best_svm=None

for learning in learning_rates:
    for regularization in regularization_strengths:
        svm=linear_classifier.Softmax()
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

y_test_pred=best_svm.predict(x_test)
test_accuracy=np.mean(y_test==y_test_pred)
print('linear svm on raw pixels final test set accuracy: %f'% test_accuracy)

w=best_svm.W[::] #1
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