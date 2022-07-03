#!/usr/bin/pyton
# -*- coding: utf-8 -*-
"""
Classification edge of deep forest algorithm
@author: Yuliang Pan

Examples: python3 deepForest.py
Note: Please change the file path(trainset, testset and resultdir)   
"""

import sys
import os
import time
import numpy as np
from sklearn.metrics import roc_auc_score
import joblib
from deepforest import CascadeForestClassifier

time_now = time.strftime("%Y%m%d%H%M%S", time.localtime())
def loadDataSet(fileName):
    fr = open(fileName, 'r')
    dataMat = []; labelMat = []
    for eachline in fr:
        lineArr = []
        curLine = eachline.strip().split(' ')  
        for i in range(2, len(curLine)-1):         
            lineArr.append(float(curLine[i]))   
        dataMat.append(lineArr)
        labelMat.append(int(curLine[-1]))      
    fr.close()
    return dataMat, labelMat
    
def splitDataSet(fileName, split_size, outdir):
    if not os.path.exists(outdir): #if not outdir,makrdir
        os.makedirs(outdir)
    fr = open(fileName,'r') #open fileName to read
    num_line = 0
    onefile = fr.readlines()
    num_line = len(onefile)  
    arr = np.arange(num_line) #get a seq and set len=numLine
    np.random.shuffle(arr) #generate a random seq from arr
    list_all = arr.tolist()
    
    each_size = int((num_line+1) / split_size) #size of each split sets
    split_all = []; each_split = []
    count_num = 0; count_split = 0  
    for i in range(len(list_all)): 
        each_split.append(onefile[int(list_all[i])].strip()) 
        count_num += 1
        if count_num == each_size:
            count_split += 1 
            array_ = np.array(each_split)
            np.savetxt(outdir + "/split_" + str(count_split) + '.txt',\
                        array_,fmt="%s", delimiter=' ')  
            split_all.append(each_split) 
            each_split = []
            count_num = 0
    return split_all

def underSample(datafile):
    dataMat,labelMat = loadDataSet(datafile) 
    pos_num = 0; pos_indexs = []; neg_indexs = []   
    for i in range(len(labelMat)): 
        if labelMat[i] == 1:
            pos_num +=1
            pos_indexs.append(i)
            continue
        neg_indexs.append(i)
    np.random.shuffle(neg_indexs)
    neg_indexs = neg_indexs[0:pos_num]
    fr = open(datafile, 'r')
    onefile = fr.readlines()
    outfile = []
    for i in range(pos_num):
        pos_line = onefile[pos_indexs[i]]    
        outfile.append(pos_line)
        neg_line= onefile[neg_indexs[i]]      
        outfile.append(neg_line)
    return outfile 

def generateDataset(datadir,outdir): 
    if not os.path.exists(outdir): #if not outdir,makrdir
        os.makedirs(outdir)
    listfile = os.listdir(datadir)
    train_all = []; test_all = [];cross_now = 0
    for eachfile1 in listfile:
        train_sets = []; test_sets = []; 
        cross_now += 1 
        for eachfile2 in listfile:
            if eachfile2 != eachfile1:
                one_sample = underSample(datadir + '/' + eachfile2)
                for i in range(len(one_sample)):
                    train_sets.append(one_sample[i])
        with open(outdir +"/test_"+str(cross_now)+".datasets",'w') as fw_test:
            with open(datadir + '/' + eachfile1, 'r') as fr_testsets:
                for each_testline in fr_testsets:                
                    test_sets.append(each_testline) 
            for oneline_test in test_sets:
                fw_test.write(oneline_test)
            test_all.append(test_sets)
        with open(outdir+"/train_"+str(cross_now)+".datasets",'w') as fw_train:
            for oneline_train in train_sets:   
                oneline_train = oneline_train
                fw_train.write(oneline_train)
            train_all.append(train_sets)
    return train_all,test_all
    
def performance(labelArr, predictArr):
    TP = 0.; TN = 0.; FP = 0.; FN = 0.   
    for i in range(len(labelArr)):
        if labelArr[i] == 1 and predictArr[i] == 1:
            TP += 1.
        if labelArr[i] == 1 and predictArr[i] == 0:
            FN += 1.
        if labelArr[i] == 0 and predictArr[i] == 1:
            FP += 1.
        if labelArr[i] == 0 and predictArr[i] == 0:
            TN += 1.
    ACC = (TP + TN) / (TP + FN + FP + TN)
    SN = TP/(TP + FN) 
    SP = TN/(FP + TN) 
    print(TP, FN, FP, TN)
    Precision = TP/(TP + FP)
    Recall = SN
    F1 = 2 * (Precision * Recall)/(Precision + Recall)
    
    fz = float(TP*TN - FP*FN)
    fm = float(np.math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    MCC = fz/fm
    return ACC, SN, SP, Precision, F1, MCC 
   
def classifier(train_X, train_y, test_X, test_y, curdir, modeldir, i):
    print (" training begin...")
    params = {'use_predictor':True,'predictor':'lightgbm','n_jobs':-1}
    clf = CascadeForestClassifier(**params)
    train_X = np.array(train_X)
    train_y = np.array(train_y)
    clf.fit(train_X,train_y)
    print (" training end.")
    # test Classifier with testsets
    print (" test begin.")
    test_X = np.array(test_X)
    joblib.dump(clf, curdir+ '/' + modeldir+ '/clf' + str(i) +'.model')
    clf = joblib.load(curdir+ '/' + modeldir+ '/clf' + str(i) +'.model')
    predict_ = clf.predict(test_X) #return type is float64
    proba = clf.predict_proba(test_X) #return type is float64
    print (" test end.") 
    ACC, SN, SP, Precision, F1, MCC = performance(test_y, predict_)
    pre_prob = []
    for i in range(len(proba)):
        pre_prob.append(proba[i][1])
    AUC = roc_auc_score(test_y, pre_prob)
    #save output 
    eval_output = []
    eval_output.append(ACC)
    eval_output.append(SN)
    eval_output.append(SP)
    eval_output.append(Precision)
    eval_output.append(F1)
    eval_output.append(MCC)
    eval_output.append(AUC)
    eval_output = np.array(eval_output, dtype=float)
    np.savetxt("proba.data",proba,fmt="%f",delimiter="\t")
    np.savetxt("test_y.data",test_y,fmt="%d",delimiter="\t")
    np.savetxt("predict.data",predict_,fmt="%d",delimiter="\t") 
    np.savetxt("eval_output.data",eval_output,fmt="%f",delimiter="\t")
    print ("Wrote results to output.data...EOF...")
    return ACC, SN, SP, Precision, F1, MCC, AUC

def mean_fun(onelist):
    count = 0
    for i in onelist:
        count += i
    return float(count/len(onelist))
    
def crossValidation(clfpath, curdir, train_all, test_all, modeldir):
    os.chdir(curdir)
    cur_path = curdir
    ACCs = [];SNs = []; SPs =[]; MCCs = []; Precisions = []; F1s = [];AUCs=[]
    for i in range(len(train_all)):
        os.chdir(cur_path)
        train_data = train_all[i]; train_X = [];train_y = []
        test_data = test_all[i];test_X = [];test_y = []
        for eachline_train in train_data:
            one_train = eachline_train.split(' ') 
            one_train_format = []
            for index in range(2, len(one_train)-1):
                one_train_format.append(float(one_train[index]))
            train_X.append(one_train_format)
            train_y.append(int(one_train[-1].strip()))
        for eachline_test in test_data:
            one_test = eachline_test.split(' ')
            one_test_format = []
            for index in range(2, len(one_test)-1):
                one_test_format.append(float(one_test[index]))
            test_X.append(one_test_format)
            test_y.append(int(one_test[-1].strip()))
        #######################################################################
        # create directory
        clfdirs = clfpath.split('/')
        parent_path = []
        for j in range(len(clfdirs)):
            clfdir = clfdirs[j]
            if j==0:
                if not os.path.exists(clfdir):
                    os.mkdir(clfdir)
                parent_path = clfdir
                continue
            current_path = str(parent_path) + '/' + str(clfdir)
            if j > 0:
                if not os.path.exists(current_path):       
                    os.mkdir(current_path) 
                parent_path = current_path
                
        out_path = clfpath + "/" + clfdirs[-1] + "_00" + str(i)  
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        if not os.path.exists(modeldir):
            os.mkdir(modeldir)
        os.chdir(out_path)
        ACC, SN, SP, Precision, F1, MCC, AUC = \
                            classifier(train_X, train_y, test_X, test_y, curdir, modeldir, i)
        #joblib.dump(clf, curdir+ '/' + modeldir+ '/clf' + str(i) +'.model')
        ACCs.append(ACC) 
        SNs.append(SN)
        SPs.append(SP)
        MCCs.append(MCC)
        Precisions.append(Precision)
        F1s.append(F1)
        AUCs.append(AUC)
        print (ACC, SN, SP, MCC, Precision, F1, AUC)
        #######################################################################
    ACC_mean = mean_fun(ACCs)
    SN_mean = mean_fun(SNs)
    SP_mean = mean_fun(SPs)
    MCC_mean = mean_fun(MCCs)
    Precision_mean = mean_fun(Precisions)
    F1_mean = mean_fun(F1s)
    AUC_mean = mean_fun(AUCs)
    ###########################################################################
    os.chdir("../")
    os.system("echo `date` '" + "' >> log.out")
    os.system("echo acc_mean=" + str(ACC_mean) + " >> log.out")
    os.system("echo sn_mean=" + str(SN_mean) + " >> log.out")
    os.system("echo sp_mean=" + str(SP_mean) + " >> log.out")
    os.system("echo precision_mean=" + str(Precision_mean) + " >> log.out")
    os.system("echo f1_mean=" + str(F1_mean) + " >> log.out")
    os.system("echo mcc_mean=" + str(MCC_mean) + " >> log.out")
    os.system("echo auc_mean=" + str(AUC_mean) + " >> log.out")
    return ACC_mean,SN_mean,SP_mean,Precision_mean,F1_mean, MCC_mean, AUC_mean

def testClf(modeldir, test_X, test_y, resultdir = "result"):
    if not os.path.exists(resultdir):
        os.mkdir(resultdir)
    # test classifier with testsets 
    predict_y = []
    Probas = np.mat(np.zeros((len(test_y),1)))
    filelist = os.listdir(modeldir) 
    count_clf  = 0
    for i in range(len(filelist)):
        if filelist[i].split('.')[1] != "model":
            continue
        count_clf += 1     
        clf = joblib.load(modeldir + '/' + filelist[i])
        print ("one cls test begining..." + str(count_clf))
        test_X = np.array(test_X)
        predict_ = clf.predict(test_X) #return type is float64
        proba = clf.predict_proba(test_X) #return type is float64
        #print proba
        np.savetxt(resultdir + "/proba" + str(count_clf) + ".data",\
                    proba,fmt="%f", delimiter="\t")
        ACC_, SN_, SP_, Precision_, F1_, MCC_ = performance(test_y, predict_)
        proba_each = []
        for i in range(len(Probas)):
            Probas[i] += proba[i][1]
            proba_each.append(proba[i][1])
        AUC_ = roc_auc_score(test_y, proba_each)
        print (ACC_, SN_, SP_, Precision_, F1_, MCC_, AUC_)
        os.system("echo ACC_mean=" + str(ACC_) +" >> "+resultdir+"/test_.log")
        os.system("echo SN_mean=" + str(SN_) + " >> "+resultdir+"/test_.log")
        os.system("echo SP_mean=" + str(SP_) +" >> "+resultdir + "/test_.log")
        os.system("echo MCC_mean=" + str(MCC_) +" >> "+resultdir+"/test_.log")
        os.system("echo Precision_mean=" + str(Precision_) +" >> " + \
                    resultdir + "/test_.log")
        os.system("echo F1_mean=" + str(F1_) +" >> "+resultdir + "/test_.log")
        os.system("echo AUC=" + str(AUC_) + " >> " + resultdir + "/test_.log")
        os.system("echo -e '\n*********************************\n' >> " + \
                    resultdir + "/test_.log")
        print ("one cls test END.")
    for i in range(len(Probas)):
        Probas[i] /=  count_clf
        if Probas[i] <= 0.5: 
            predict_y.append(0)
        else:
            predict_y.append(1)
    ACC, SN, SP, Precision, F1, MCC = performance(test_y, predict_y) 
    AUC = roc_auc_score(test_y, Probas)
    np.savetxt(resultdir + "/proba_avg.data",Probas,fmt="%f",delimiter="\t")
    np.savetxt(resultdir+"/predict_avg.data",predict_y,fmt="%d",delimiter="\t")  
    os.system("echo `date` '"+str(filelist)+"' >> result.data")
    os.system("echo ACC_mean=" + str(ACC) +" >>  result.data")
    os.system("echo SN_mean=" + str(SN) + " >> result.data")
    os.system("echo SP_mean=" + str(SP) +" >> result.data")
    os.system("echo Precision_mean=" + str(Precision) +" >> result.data")
    os.system("echo F1_mean=" + str(F1) +" >> result.data")
    os.system("echo MCC_mean=" + str(MCC) +" >> result.data")
    os.system("echo AUC=" + str(AUC) + " >> result.data")
    return ACC, SN, SP, Precision, F1, MCC, AUC

def CV_train(basedir, trainsets, k, splitdir, sampledir, clfpath, modeldir):
    os.chdir(basedir)
    print ("split datasets start...")  
    splitDataSet(trainsets, k, splitdir) 
    print ("End of split, starting to generateDatasets...") 
    datadir = splitdir; 
    outdir = sampledir
    train_all, test_all = generateDataset(datadir, outdir)    
    print ("End of generateDataset  and cross validation start")
    curdir = basedir
    ACC_mean, SN_mean, SP_mean, Precision_mean, F1_mean, MCC_mean, AUC_mean = \
            crossValidation(clfpath, curdir, train_all,test_all, modeldir)
    print (ACC_mean, SN_mean, SP_mean, Precision_mean, F1_mean,MCC_mean,AUC_mean)


def CV_GTB_train(basedir, trainsets, k, splitdir, sampledir, clfpath, \
                modeldir, testset):
    CV_train(basedir, trainsets, k, splitdir, sampledir, clfpath, modeldir)

def CV_test(testset, clfpath, modeldir):
    test_X, test_y = loadDataSet(testset)
    os.chdir(clfpath)
    ACC, SN, SP, Precision, F1, MCC, AUC = testClf(modeldir, test_X, test_y)
    print (ACC, SN, SP, Precision, F1, MCC, AUC)

def train_gtb_model(trainsets, k, result_path, desc, testset):
    basedir = result_path
    if not os.path.exists(basedir):
        os.mkdir(basedir)
    splitdir = 'split' + str(k) + "_" + desc
    sampledir = "sample" + "_" + desc
    clfpath = "CV" + "_" + desc
    modeldir = clfpath + "/model"
    CV_GTB_train(basedir, trainsets, k, splitdir, sampledir, clfpath, \
                    modeldir, testset)
    clfdir = basedir + "/" + clfpath
    return clfdir

def RunPred(trainset, testset, result_path, k=10, desc="DF"):
    clfdir = train_gtb_model(trainset, k, result_path, desc, testset)
    modeldir = clfdir + "/model"
    CV_test(testset, clfdir, modeldir)

if __name__ == '__main__':
    
    #train model
    trainset = "train.txt"   #Training set path
    testset = "test.txt"     #testing set path
    resultdir = "classification_model"   #Model storage path
    RunPred(trainset, testset, resultdir)
    
    '''
    #test model
    testset = "/data/panyuliang/dataSource/combineFeature/Dataset/Scale_Data/otherFile.txt"  #unlabel set path
    clfdir = "/data/panyuliang/classification_methods/1/CV_DF"                               #model folder
    modeldir = "/data/panyuliang/classification_methods/1/CV_DF/model"                       #model path
    CV_test(testset, clfdir, modeldir)
    '''
    

