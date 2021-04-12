# -*- coding: utf-8 -*-
"""
@author: Yaya Liu

Objectives:
- Files partition based on number of folds, and for each fold, save test file names for later use
- Create training files for each fold, and these training files will be used to generate new fable models for each fold

2. How to run it
- Make sure you can run fable demo - demo.py under a Linux system or Linux-like system
- Create file folder "trainning_build_data" under" the same directory with fable model
- Make sure all raw text files and annotation files are put under the folder: "input_data" 
- Run this file under a Linux system or Linux-like system. Output will be generated under folder "trainning_build_data"

"""

import numpy as np
import glob
from sklearn.utils import shuffle
import fable, utils

def prepareTrainFiles(file):  
    """
    Prepare training files by using part of fable code
    Input: Raw medical records
    Output: prepared files that stored in folder: "\trainning_build_data"
    """
    
    print("Fable preparing training files...", file)       
    fable.fable(file, 1, -1)  # prepare training files using fable


def filePartitions(flist, folds):
    """
    Files partition based on number of folds
    Input: all prepared training files
    Output: 2 nested list. 
            test_fold - file names for test in each fold
            train_fold - file names for training in each fold
    """
    
    num_val_samples = len(flist) // folds 
    
    test_fold = []
    train_fold = []
    
    for i in range(folds):
        test_fold_data = flist[i * num_val_samples : (i + 1) * num_val_samples] # prepare the test data
        test_fold.append(test_fold_data)
         
        train_fold_data = np.concatenate([flist[:i * num_val_samples], flist[(i + 1) * num_val_samples:]], axis = 0).tolist() # prepares the training data
        train_fold.append(train_fold_data)
    return test_fold, train_fold

     
def addBIOlable(files, i):
    """
    For prepared training files, add each token's BIO lable which is extracted from annotation files
    Input: traing files of each fold, the index of current fold
    Output: one finalized training file for each fold. 
    """
        
    print("Adding BIO labels for folder", i) 
            
    global fRawPath, fPath
    
    fName = "temp_" + str(i)
    trainFile = open(fPath + fName, 'w')      
    
    for file in files:         
        BIOlist = [] 
        mappingDict_char, mappingDict_sent = {}, {}
        BIOSentWordDict = {} 
               
        fileName = file.split('/')[2].split('.')[0]
        BIOlist = utils.BIOAnnfile(fRawPath + fileName + '.ann')   
        content = utils.rawTextPreprocess(fRawPath + fileName + '.txt')            
        mappingDict_char, mappingDict_sent = utils.rawMappingDict(content)
        BIOSentWordDict = utils.BIOSentWordIndex(BIOlist, mappingDict_char, fRawPath + fileName + '.txt')
#        print(BIOSentWordDict)         
              
        with open(file) as f:
            sents = f.read().split('\n')            
            for sent in sents:
                if sent != '':
                    tokens = sent.split('\t')
                    if (int(tokens[-2]), int(tokens[-1])) in BIOSentWordDict: 
                        tokens[0] = BIOSentWordDict[(int(tokens[-2]), int(tokens[-1]))][0]
                        #trainFile.write('\t'.join(tokens[0:-2]))  
                        trainFile.write('\t'.join(tokens[0:-2])) 
                        trainFile.write('\n')
                    else:
                        trainFile.write('\t'.join(tokens[0:-2])) 
                        trainFile.write('\n')                                                            
                else:
                    trainFile.write('\n')         
    trainFile.close() 

fRawPath = './input_data/'     
fPath = './trainning_build_data/'

def main():
    rawlist = glob.glob(fRawPath + '*.txt', recursive = True)        
    for file in rawlist: 
        prepareTrainFiles(file)  

    folds = 4    # 4-fold cross validation    
    flist = glob.glob(fPath + '*.train', recursive = True) 
    flist = shuffle(flist, random_state = 2021)
        
    test_fold, train_fold = filePartitions(flist, folds)    
    
    for i in range(folds):
        addBIOlable(train_fold[i], i)
    
#        ftrainName = "train_" + str(i)            # save the name of training files for each fold
#        trainFile = open(fPath + ftrainName, 'w')        
#        for item in train_fold[i]:
#            trainFile.writelines(item + '\n')        
#        trainFile.close()        
        
        ftestName = "test_" + str(i)             #save the name of test files for each fold
        testFile = open(fPath + ftestName, 'w')
        for item in test_fold[i]:
            testFile.writelines(item + '\n')        
        testFile.close()
                
    
if __name__ == "__main__":
    main()