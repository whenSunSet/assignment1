import pickle
import numpy as np
import os
import cs231n.classifiers.features as features

def load_cifar_batch(filename):
    with open(filename,'rb') as f :
        datadict=pickle.load(f,encoding='bytes')
        x=datadict[b'data']
        y=datadict[b'labels']
        x=x.reshape(10000,3,32,32).transpose(0,2,3,1).astype('float')
        y=np.array(y)
        return x,y

def load_cifar10(root):
    xs=[]
    ys=[]
    for b in range(1,6):
        f=os.path.join(root,'data_batch_%d' % (b,))
        x,y=load_cifar_batch(f)
        xs.append(x)
        ys.append(y)
    Xtrain=np.concatenate(xs)
    Ytrain=np.concatenate(ys)
    del x ,y
    Xtest,Ytest=load_cifar_batch(os.path.join(root,'test_batch'))
    return Xtrain,Ytrain,Xtest,Ytest

def init_train_data(data_file='../datasets/cifar-10-batches-py' , num_training=49000 , num_validation=1000 , num_test=1000 , num_dev=500):
    x_train,y_train,x_test,y_test=load_cifar10(data_file)

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

    # 归一化处理
    mean_image=np.mean(x_train,axis=0)
    x_train-=mean_image
    x_val-=mean_image
    x_test-=mean_image
    x_dev-=mean_image
    return x_train , y_train , x_val , y_val , x_test , y_test , x_dev , y_dev

def init_train_feature_data():
    x_train, y_train , x_val , y_val, x_test , y_test , x_dev , y_dev = init_train_data()
    num_color_bins = 10
    feature_fns = [features.hog_feature, lambda img: features.color_histogram_hsv(img,nbin=num_color_bins)]

    X_train_feats = features.extract_features(x_train, feature_fns, verbose=True)
    X_val_feats = features.extract_features(x_val, feature_fns)
    X_test_feats = features.extract_features(x_test, feature_fns)
    X_dev_feats = features.extract_features(x_dev, feature_fns)

    mean_feat=np.mean(X_train_feats,axis=0,keepdims=True)
    X_train_feats-=mean_feat
    X_val_feats-=mean_feat
    X_test_feats-=mean_feat
    X_dev_feats-=mean_feat

    std_feat=np.std(X_train_feats,axis=0,keepdims=True)
    X_train_feats/=std_feat
    X_val_feats/=std_feat
    X_test_feats/=std_feat
    X_dev_feats/=std_feat

    X_train_feats=np.hstack([X_train_feats,np.ones((X_train_feats.shape[0],1))])
    X_val_feats=np.hstack([X_val_feats,np.ones((X_val_feats.shape[0],1))])
    X_test_feats=np.hstack([X_test_feats,np.ones((X_test_feats.shape[0],1))])
    X_dev_feats=np.hstack([X_dev_feats,np.ones((X_dev_feats.shape[0],1))])

    return X_train_feats, y_train , X_val_feats , y_val, X_test_feats , y_test , X_dev_feats , y_dev