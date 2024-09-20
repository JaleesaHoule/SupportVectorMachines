import numpy as np
import scipy 
import cv2
import pandas as pd
from libsvm.svmutil import *
from sklearn.preprocessing import MinMaxScaler



def read_images(directory1, directory2):
    images = []
    image_ids = []
    for i in os.listdir(directory1):
        if i.endswith('.tif'):
            images.append(cv2.imread(directory1+i, cv2.IMREAD_GRAYSCALE))
    for i in os.listdir(directory2):
        if i.endswith('.tif'):
            images.append(cv2.imread(directory2+i, cv2.IMREAD_GRAYSCALE))
                    
    return images 

def get_optimum_params(misclassifications, kernel_summaries):
    averages = np.sum(misclassifications, axis=0)/len(misclassifications)
    df = pd.DataFrame({"Params" : kernel_summaries[0].flatten(), "Fold1" : misclassifications[0], "Fold2" : misclassifications[1], "Fold3" : misclassifications[2], "Average" : averages})
    best_loc = np.where(averages==np.min(averages))[-1]
    best_params=[]
    for i,j in enumerate(best_loc):
        best_params.append(kernel_summaries[0][j][0].split(", Classification")[0])
    
    print('Best average error: ', averages[best_loc])
    print('Best parameters: ', best_params)
    return df
    
def run_SVM(traindata, trainlabels, testdata, testlabels, params):
    accuracy=[]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_traindata = scaler.fit_transform(traindata)
    scaled_testdata = scaler.fit_transform(testdata)
    model = svm_train(trainlabels, scaled_traindata, params)
    predictions, stats, vals  = svm_predict(testlabels, scaled_testdata, model)
    accuracy.append(100-stats[0])
    return np.array(accuracy)  

def train_model(traindata, trainlabels, testdata, testlabels, C_options=[0.1,1,10,100], kernel='poly'):
    
    # transform data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_traindata = scaler.fit_transform(traindata)
    scaled_testdata = scaler.fit_transform(testdata)
    
    ## equivalent method using lib-svm: 
    ### x = scipy.sparse.csr_matrix(traindata)
    ### scale_param = csr_find_scale_param(x)
    ### scaled_x = csr_scale(x, scale_param)


    if kernel=='poly':
        d=[1,2,3]
        info=[]
        accuracy=[]
        for j in d:
            for i in C_options:
                #print('------------------------------------------')
                params = '-q -s 0 -t 1 -d ' + str(j) + ' -g 1 -r 0 -c ' + str(i)
                #print('\n \n Polynomial Kernel, d=', j, 'Cost=', i, '\n')
                model = svm_train(trainlabels, scaled_traindata, params)
                #print('\n')
                predictions, stats, vals = svm_predict(testlabels, scaled_testdata, model, '-q')
                string = 'Polynomial, d= ' + str(j) + ', C= ' + str(i) # + ", Classification accuracy = "  + str(stats[0])
                info.append([string])
                accuracy.append(100-stats[0])
    
    if kernel=='RBF':
        gamma = [0.1,1,10,100]
        info=[]
        accuracy = []
        for j in gamma:
            for i in C_options:
                #print('------------------------------------------')
                params = '-q -s 0 -t 2 -g ' + str(j) + ' -c ' + str(i)
                #print('\n \n RBF Kernel, g=', j, 'Cost=', i, '\n')
                model = svm_train(trainlabels, scaled_traindata, params)
                #print('\n')
                predictions, stats, vals  = svm_predict(testlabels, scaled_testdata, model, '-q')
                string = 'RBF, gamma= ' + str(j) + ', C= ' + str(i) # + ", Classification accuracy = " + str(stats[0])
                info.append([string])
                accuracy.append(100-stats[0])

                    
    return info, np.array(accuracy)    
