import os
from turtle import right
os.environ['CUDA_VISIBLE_DEVICES']  = '0'
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import h5py


def post_process(model, output_dir):

    img = np.load('02_scan_x256_y256_raw.npy')
    img = np.transpose(img,(2,3,0,1))
    data_r = np.copy(img)
    data_r[data_r>1e3]=1e3
    min_ = np.min(data_r)
    max_ = np.max(data_r)

    data_r = 1.0*(data_r-min_)/(max_-min_)

    data_r = data_r.reshape(-1,1,124,124)
    data_r_cut = data_r[:,:,2:122,2:122]
    data_r_cut = data_r_cut.reshape(256,256,120,120)
    X = data_r_cut.reshape(-1, 120,120)

    # generate outputs/targets
    dataset_h5 = h5py.File('unbinned_results.h5','r+')
    rots = np.array(dataset_h5['rotation'])
    scal = np.array(dataset_h5['scale'])
    y = np.concatenate((rots, scal), axis=1)
    sc = StandardScaler()
    _ = sc.fit(y)
    base_8 = np.load('base_8.npy', allow_pickle=1)


    predictions = model.predict(np.ascontiguousarray(X))
    predictions = sc.inverse_transform(predictions)
    # print(sc.inverse_transform(model.predict(X[0])))
    # print(y[0])

    rotation_, scale_ = predictions[:, 0:2], predictions[:, 2:]

    mse = np.mean((predictions - y)**2)
    print('model MSE on full dataset: ', mse)

    #base=2,loss=0.024
    j=0
    list_new = []
    fig,ax = plt.subplots(1,3,figsize=(20,10))
    for i in range(2):
        if np.sum(base_8[:,i]!=0):
            j+=1
            print(i)
            list_new.append(i)
            ax[i].title.set_text(str(i))
            ax[i].imshow(base_8[:,i].reshape(256,256))
    print('total activated channels: '+str(j))
    ax[2].title.set_text('mean of the sample domain')
    ax[2].imshow(np.mean(data_r_cut.reshape(256,256,-1),axis=2))

    fig,ax = plt.subplots(2,2,figsize = (10,10))
    ax[0][0].imshow(rotation_[:,0].reshape(256,256))
    ax[0][1].imshow(rotation_[:,1].reshape(256,256))
    ax[1][0].hist(rotation_[:,0].reshape(-1),200)
    ax[1][1].hist(rotation_[:,1].reshape(-1),200)
    fig.savefig('{output_dir}/rotation.png'.format(output_dir=output_dir))

    sample_base = base_8[:,1].reshape(256,256)

    scale_0 = np.multiply(sample_base.reshape(256,256),scale_[:,0].reshape(256,256))
    scale_1 = np.multiply(sample_base.reshape(256,256),scale_[:,3].reshape(256,256))

    fig,ax = plt.subplots(2,2,figsize = (10,10))
    clim0 = [1.14,1.19]
    clim1 = [1.1,1.17]
    ax[0][0].imshow(scale_0,clim=clim0)
    ax[0][1].imshow(scale_1,clim=clim1)
    ax[1][0].hist(scale_0.reshape(-1),200,range=clim0)
    ax[1][1].hist(scale_1.reshape(-1),200,range=clim1)
    fig.savefig('{output_dir}/scale.png'.format(output_dir=output_dir))


    right_tri = np.sqrt(scale_0**2+scale_1**2).reshape(256,256)
    np.save('right_triangle_mlp_average_pool_12b.npy', right_tri)
    fig,ax = plt.subplots(1,2,figsize=(20,10))
    clim=[1.59,1.65]
    ax[0].imshow(right_tri,clim=clim)
    ax[1].hist(right_tri.reshape(-1),200,range=clim)
    fig.savefig('{output_dir}/right_triangle.png'.format(output_dir=output_dir))



