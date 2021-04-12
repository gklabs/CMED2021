# -*- coding: utf-8 -*-
"""
@author: Yaya Liu

Objective: evaluate falel's performance on token and entity level

"""

# dictionary for label transformation

#lookup = {        
#            'M': 'Drug',
#            'DO': 'Dosage',   
#            'F': 'Frequency',
#            'MO': 'Route',
#            'R': 'Reason',                        
#            'DU': 'Duration'        
#        }

import re
import utils
import logging
from string import punctuation
import fable_predict, fable_parse


def parsePredict_token(file):
    '''
    Parse the output of fable
    Input: files with predictions on tokens, output of fable_predict.py
    Output: a nested list - [[BIOlabel of the token, token, sentIndex, wordIndex]]
    '''
       
    predOutput_token = []
    with open(file, 'r') as rd:
        for line in rd:
            line = line.strip()      # strip the last '\n'                
            BIOlabel = line.split('|')[0]
            if BIOlabel != 'NO':
                word =  line.split('|')[1]           
                s = re.sub(r'[^\w\s]','', word)   # replace all punctuations in the medication names with ''        
                if BIOlabel.split('-')[1] == 'M' and len(s) != 0:  # fliter out medications, and the medications are not only consist of punctuations
                    sentIndex = line.split('|')[2]
                    wordIndex = line.split('|')[3]
                    predOutput_token.append([BIOlabel, word, sentIndex, wordIndex])
    #print(predOutput_token)
    return predOutput_token


#log = "fableEvalLogging.txt"   # create a log file
#logging.basicConfig(filename = log, level = logging.DEBUG, format = '%(message)s')  


def predEval_token(files, currFold):     
    '''
    Evaluate fable's performance on token level
    Input: - files with predictions on tokens, output of fable_predict.py
           - fold index
    Output: 3 lists used to store F1, recall, precision of each fold (on token level)
    '''
    global fRawPath 
    global f1List_token, recallList_token, precisionList_token
        
    correctPred = 0
    wrongBIO = 0
    wrongToken = 0
    wrongFind = 0
    totalMedTokens = 0  # total mecical tokens in annotation files
    
    for file in files:
        print()
        print("Evaluating On Token Level...", file)    
        logging.info("         ")
        logging.info("Evaluating On Token Level %s...", file)
        
        predOutput_token = []        
        BIOlist = []
        mappingDict_char, mappingDict_sent = {}, {}
        BIOSentWordDict = {} 
                
        predOutput_token = parsePredict_token(file)
        #print(predOutput)
        #print()
        
        fileName = file.split('/')[2].split('.')[0] + '.ann'
        BIOlist = utils.BIOAnnfile(fRawPath + fileName)                   
        totalMedTokens += len(BIOlist)
        #BIOlist = utils.BIOAnnfile('./input_data/322-04.ann')   
        #print(BIOlist)   
        #print()
        
        rawfile = file.split('/')[2].split('.')[0] + '.txt'
        content = utils.rawTextPreprocess(fRawPath + rawfile)
        mappingDict_char, mappingDict_sent = utils.rawMappingDict(content)
        #mappingDict = utils.rawMappingDict('./input_data/136-02.txt')
        
        BIOSentWordDict = utils.BIOSentWordIndex(BIOlist, mappingDict_char, file)
        #print(BIOSentWordDict)

        for i in range(len(predOutput_token)):
            tmp = predOutput_token[i]
            if (int(tmp[2]), int(tmp[3])) in BIOSentWordDict:   # check (sentence_index, word_index)
                if tmp[0] == BIOSentWordDict[(int(tmp[2]), int(tmp[3]))][0]:   # check BIO label  
                    pred_token = tmp[1]
                    if pred_token[-1] in punctuation:
                        pred_token = pred_token[0:-1]
                    if pred_token == BIOSentWordDict[(int(tmp[2]), int(tmp[3]))][1]:
                        correctPred += 1
                    else:
                        print("Token not matched", tmp, BIOSentWordDict[(int(tmp[2]), int(tmp[3]))])
                        logging.info('Token not matched: %s, %s', tmp, BIOSentWordDict[(int(tmp[2]), int(tmp[3]))])    
                        wrongToken += 1                        
                else:       
                    print("BIO label not matched", tmp, BIOSentWordDict[(int(tmp[2]), int(tmp[3]))])
                    logging.info('BIO label not matched: %s, %s', tmp, BIOSentWordDict[(int(tmp[2]), int(tmp[3]))])    
                    wrongBIO += 1
            else:
                print("Cannot find in Annotation files: ", tmp)
                logging.info('Cannot find in Annotation files: %s', tmp) 
                wrongFind += 1
                   
    recall = round(correctPred/totalMedTokens, 4)    
    precision = round(correctPred/(correctPred + wrongBIO + wrongToken + wrongFind), 4)
    f1 = round(2 * recall * precision/(recall + precision), 4)
    
    print()
    print("On Token Level: Results of folder", currFold)
    print('f1: ', f1)  
    print('recall: ', recall)  
    print('precision: ', precision)        
    
    print("Correct predicted cases: ", correctPred)
    print("BIO label not matched cases: ", wrongBIO)
    print("Token not matched cases: ", wrongToken)
    print("Cannot find in Annotation files cases: ", wrongFind)
    print("Total recognized medication: ", correctPred + wrongBIO + wrongToken + wrongFind)
    print("Total annotated medication: ", totalMedTokens)
    
    logging.info("         ")
    logging.info("On Token Level: Results of folder %d", currFold)
    logging.info('f1: %f', f1)   
    logging.info('recall: %f', recall)  
    logging.info('precision: %f', precision)        
   
    logging.info('Correct predicted cases: %d', correctPred)
    logging.info('BIO label not matched cases: %d', wrongBIO)
    logging.info("Token not matched cases: %d", wrongToken)    
    logging.info('Cannot find in Annotation files cases: %d', wrongFind)  
    logging.info('Total recognized tokens: %d', correctPred + wrongBIO + wrongToken + wrongFind) 
    logging.info('Total annotated tokens: %d', totalMedTokens) 
    
    f1List_token.append(f1)
    recallList_token.append(recall)
    precisionList_token.append(precision)


def predEval_entity(files, currFold):   
    '''
    Evaluate fable's performance on entity level
    Input: - files with predictions on tokens, output of fable_predict.py
           - fold index
    Output: 3 lists used to store F1, recall, precision of each fold (on entity level)
    '''
    
    global fRawPath 
    global f1List_entity, recallList_tentity, precisionList_entity
        
    correctPred = 0
    wrongEntity = 0
    wrongEndIndex = 0
    wrongFind = 0
    totalEntity = 0  # total mecical tokens in annotation files
    
    for file in files:
        print()
        print("Evaluating On Entity Level...", file)    
        logging.info("         ")
        logging.info("Evaluating On Entity Level %s...", file)   
                
        EntityDict = {} 

        fileName = file.split('/')[2].split('.')[0] + '.ann'
        EntityDict = utils.EntityAnnfile(fRawPath + fileName)
        #print(EntityDict)
        totalEntity += len(EntityDict)
        
        predOutput_entity = []        
        predOutput_entity = fable_parse.parse(file)        
        for i in range(len(predOutput_entity)):            
            starIndex = str(predOutput_entity[i][0])
            endIndex = str(predOutput_entity[i][1])
            predEntity = str(predOutput_entity[i][2])
            
            if starIndex in EntityDict:
                if endIndex == EntityDict[starIndex][0]:
                    if predEntity == EntityDict[starIndex][1]:
                        correctPred += 1                       
                    else:
                        print("Entity not matched", predOutput_entity[i], EntityDict[starIndex])
                        logging.info('Entity not matched: %s, %s', predOutput_entity[i],  EntityDict[starIndex])    
                        wrongEntity += 1                                        
                else:
                    print("End Index not matched", predOutput_entity[i], EntityDict[starIndex])
                    logging.info('End Index not matched %s, %s', predOutput_entity[i],  EntityDict[starIndex])    
                    wrongEndIndex += 1                                       
            else:
                print("Cannot find this entity in Annotation files: ", predOutput_entity[i])
                logging.info('Cannot find this entity in Annotation files: %s', predOutput_entity[i]) 
                wrongFind += 1
        
    recall = round(correctPred/totalEntity, 4)    
    precision = round(correctPred/(correctPred + wrongEntity + wrongEndIndex + wrongFind), 4)
    f1 = round(2 * recall * precision/(recall + precision), 4)
    
    print()
    print("On Entity Level: Results of folder", currFold)
    print('f1: ', f1)  
    print('recall: ', recall)  
    print('precision: ', precision)        
    
    print("Correct predicted cases: ", correctPred)
    print("Entity not matched cases: ", wrongEntity)
    print("End index not matched cases: ", wrongEndIndex)
    print("Cannot find in Annotation files cases: ", wrongFind)
    print("Total recognized medication: ", correctPred + wrongEntity + wrongEndIndex + wrongFind)
    print("Total annotated medication: ", totalEntity)
    
    
    logging.info("         ")
    logging.info("On Entity Level: Results of folder %d", currFold)
    logging.info('f1: %f', f1)   
    logging.info('recall: %f', recall)  
    logging.info('precision: %f', precision)        
   
    logging.info('Correct predicted cases: %d', correctPred)
    logging.info('Entity not matched cases: %d', wrongEntity)
    logging.info("End index not matched cases: %d", wrongEndIndex)    
    logging.info('Cannot find in Annotation files cases: %d', wrongFind)  
    logging.info('Total recognized medication: %d', correctPred + wrongEntity + wrongEndIndex + wrongFind) 
    logging.info('Total annotated medication: %d', totalEntity) 
    
    f1List_entity.append(f1)
    recallList_entity.append(recall)
    precisionList_entity.append(precision)        
        
          
      
fRawPath = './input_data/'     
fPath = './trainning_build_data/'
fPredPath = './predict_data/'
  
f1List_token, recallList_token, precisionList_token = [], [], [] 
f1List_entity, recallList_entity, precisionList_entity = [], [], [] 
    
def main():   
    folds = 4    # 4-fold cross validation  
    for i in range(folds): 
       testList, predList = [], []
       fTestName = "test_" + str(i)   
                       
       with open(fPath + fTestName, 'r') as rd:         # get the test file list for each folds
           for line in rd:
               line = line.rstrip('\n')
               fRawName = line.split('/')[2].split('.')[0] + '.txt'
               testList.append(fRawPath + fRawName)
               predList.append(fPredPath + fRawName)
                   
       print()        
       print("Processing fold", i)  
       logging.info("         ")
       logging.info("************* Processing fold %d *************", i)
        
       for j in range(len(testList)):
           fable_predict.predict(testList[j], 0, i)       # call fable to predict BIO labels for each token
       
           
       predEval_token(predList, i)
       predEval_entity(predList, i)
       
    print()
    print("********* On Token Level: CV Final Results *********************")
    print("Cross-validation F1 list", f1List_token)  
    print("Cross-validation recall list", recallList_token) 
    print("Cross-validation precision list", precisionList_token)
    
    
    f1AVG_token = round(sum(f1List_token)/len(f1List_token), 4)
    recallAVG_token = round(sum(recallList_token)/len(recallList_token), 4)
    precisionAVG_token = round(sum(precisionList_token)/len(precisionList_token), 4)
    
    print()
    print("Cross-validation average F1", f1AVG_token)  
    print("Cross-validation average recall", recallAVG_token) 
    print("Cross-validation average precision", precisionAVG_token)

    logging.info("         ")
    logging.info("********* On Token Level: CV Final Results   *********************")
    logging.info("Cross-validation F1 list: %s", f1List_token)  
    logging.info("Cross-validation recall list: %s", recallList_token) 
    logging.info("Cross-validation precision list: %s", precisionList_token)
    logging.info("         ")
    logging.info('Cross-validation average F1: %f', f1AVG_token)  
    logging.info('Cross-validation average recall: %f', recallAVG_token)  
    logging.info('Cross-validation average precision: %f', precisionAVG_token)
    
    
    print()
    print("********* On Entity Level: CV Final Results *********************")
    print("Cross-validation F1 list", f1List_entity)  
    print("Cross-validation recall list", recallList_entity) 
    print("Cross-validation precision list", precisionList_entity)
    
    
    f1AVG_entity = round(sum(f1List_entity)/len(f1List_entity), 4)
    recallAVG_entity = round(sum(recallList_entity)/len(recallList_entity), 4)
    precisionAVG_entity = round(sum(precisionList_entity)/len(precisionList_entity), 4)
    
    print()
    print("Cross-validation average F1", f1AVG_entity)  
    print("Cross-validation average recall", recallAVG_entity) 
    print("Cross-validation average precision", precisionAVG_entity)

    logging.info("         ")
    logging.info("********* On Entity Level: CV Final Results   *********************")
    logging.info("Cross-validation F1 list: %s", f1List_entity)  
    logging.info("Cross-validation recall list: %s", recallList_entity) 
    logging.info("Cross-validation precision list: %s", precisionList_entity)
    logging.info("         ")
    logging.info('Cross-validation average F1: %f', f1AVG_entity)  
    logging.info('Cross-validation average recall: %f', recallAVG_entity)  
    logging.info('Cross-validation average precision: %f', precisionAVG_entity)    
    

        
                  
if __name__ == "__main__":
    main()
    
    
    