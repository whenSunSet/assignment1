import numpy as np
import matplotlib.pyplot as plt
import cs231n.classifiers.neural_net as neural_net
import cs231n.classifiers.vis_utils as vis_utils

from cs231n.classifiers.data_util import init_train_data
x_train , y_train , x_val , y_val , x_test , y_test , x_dev , y_dev = init_train_data()

input_size=32*32*3
num_classes=10
results={}
best_val_acc=0
best_net=None

# hidden_size=[75,100,125]
# learning_rates=np.array([0.7,0.8,0.9,1.0,1.1])*1e-3
# regularization_strengths=[0.75,1.0,1.25]

hidden_size=[100]
learning_rates=np.array([1.0])*1e-3
regularization_strengths=[1.0]

for hs in hidden_size:
    for lr in learning_rates:
        for reg in regularization_strengths:
            net=neural_net.TwoLayerNet(input_size,hs,num_classes)
            stats=net.train(x_train,y_train,x_val,y_val,num_iters=1500,batch_size=256,
                            learning_rate=lr,learning_rate_decay=0.95,reg=reg,verbose=False)
            val_acc=(net.predict(x_val)==y_val).mean()
            if val_acc >best_val_acc:
                best_val_acc=val_acc
                best_net=net
            results[(hs,lr,reg)]=val_acc

for hs,lr,reg in sorted(results):
    val_acc=results[(hs,lr,reg)]
    print('hs %d lr %e reg %e val accuracy: %f' % (hs,lr,reg,val_acc))
print('best validation accuracy achieved during cross_validation: %f' %
      best_val_acc)

test_acc=(best_net.predict(x_test)==y_test).mean()
print('test accuracy:' , test_acc)


vis_utils.show_net_weights(best_net)