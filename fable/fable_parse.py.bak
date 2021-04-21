# -*- coding: utf-8 -*-
"""
@author: Yaya Liu

1. This scripts interprets outputs(sequence labeling) from "fable_predict.py"

2. The output is a nested list
[start index of the medication entity, end index of the medication entity, medication entity]

example:
[[345, 353, 'ATENOLOL'],
[407, 416, 'CAPTOPRIL'],
[470, 477, 'ASPIRIN'],
[479, 499, 'ACETYLSALICYLIC ACID'],
[863, 872, 'captopril']]

    
3. How to run it
- Create file folder "parse_data" under the same directory with fable model
- Run fable_predict.py under a Linux system or Linux-like system
- Run this file under a Linux system or Linux-like system. The output will be generated under folder "parse_data"

"""

import os, glob
import re
from string import punctuation
import utils


# dictionary for label transformation
#lookup = {        
#            'M': 'Drug',
#            'DO': 'Dosage',   
#            'F': 'Frequency',
#            'MO': 'Route',
#            'R': 'Reason',                        
#            'DU': 'Duration'        
#        }


def checkPunc(word):
    '''
    Check whether the last character of a word is a punctuation
    Input: word
    Output: boolean yes or no      
    '''
    
    if word[-1] in punctuation:
        return True
    else:
        return False

def parse(file):
    '''
    Ensemble recognized tokens to entity based on BIO lables
    Input: files with predictions on tokens, output of fable_predict.py
    Output: a nested list - [[start index of the medication entity, end index of the medication entity, medication entity]]    
    '''
    
    out = []
    predOutput_entity = []
    mappingDict_char, mappingDict_sent = {}, {}
    
    rawfile = file.split('/')[2].split('.')[0] + '.txt'
    content = utils.rawTextPreprocess(fRawPath + rawfile)
    mappingDict_char, mappingDict_sent = utils.rawMappingDict(content)
    #print(mappingDict_sent)

    with open(file, 'r') as rd:
        for line in rd:
            line = line.strip()      # strip the last '\n'                
            BIOlabel = line.split('|')[0]
            if BIOlabel != 'NO':
                word =  line.split('|')[1]           
                s = re.sub(r'[^\w\s]','', word)   # replace all punctuations in the medication names with ''  
                if BIOlabel.split('-')[1] == 'M' and len(s) != 0:  # fliter out medications, and the medications are not only consist of punctuations
                    out.append(line)            
    #print(out)
        
    predOutput_entity = []
    tmp = '' 
    tmpIndex = []  # store the start and end of character index
    for i in range(len(out)):
        #curr_BIO_position = out[i].split('|')[0].split('-')[0]   # label position
        curr_BIO = out[i].split('|')[0].split('-')[1]             # BIO lable, such as M/DO  
        curr_word = out[i].split('|')[1]                          # previous word
        sent_index = out[i].split('|')[2]
        word_index = out[i].split('|')[3]
        tmp += curr_word + ' '  
        
        tmpTuple = (int(sent_index), int(word_index))
        charStart = mappingDict_sent[tmpTuple][0]
        charEnd = mappingDict_sent[tmpTuple][1]              
        tmpIndex.append(charStart)
        tmpIndex.append(charEnd)        

        if i == len(out) - 1:
            #print("111111")
            if checkPunc(tmp.rstrip()[-1]):
                tmp = tmp.rstrip()[0:-1]
                predOutput_entity.append([tmpIndex[0], tmpIndex[-1] - 1, tmp])
            else:
                predOutput_entity.append([tmpIndex[0], tmpIndex[-1], tmp.rstrip()])
            tmp = ''
            tmpIndex = []
        else:
            next_BIO_position = out[i + 1].split('|')[0].split('-')[0]   # label position: B or I
            next_BIO = out[i + 1].split('|')[0].split('-')[1]   # BIO label position: M or others
            #print('next_position', out[i+1], next_position)                
            if next_BIO_position == 'B':  
                #print("222222")
                if checkPunc(tmp.rstrip()[-1]):
                    tmp = tmp.rstrip()[0:-1]
                    predOutput_entity.append([tmpIndex[0], tmpIndex[-1] - 1, tmp])
                else:
                    predOutput_entity.append([tmpIndex[0], tmpIndex[-1], tmp.rstrip()])
                tmp = ''
                tmpIndex = []
            elif next_BIO_position == 'I':
                if sent_index != out[i + 1].split('|')[2]:       # not in the same sentence   
                    #print("333333333")
                    if checkPunc(tmp.rstrip()[-1]):
                        tmp = tmp.rstrip()[0:-1]
                        predOutput_entity.append([tmpIndex[0], tmpIndex[-1] - 1, tmp])
                    else:
                        predOutput_entity.append([tmpIndex[0], tmpIndex[-1], tmp.rstrip()])
                    tmp = ''
                    tmpIndex = []
                else:
                    if next_BIO != curr_BIO:
                        #print("4444444444")
                        if checkPunc(tmp.rstrip()[-1]):
                            tmp = tmp.rstrip()[0:-1]
                            predOutput_entity.append([tmpIndex[0], tmpIndex[-1] - 1, tmp])
                        else:
                            predOutput_entity.append([tmpIndex[0], tmpIndex[-1], tmp.rstrip()])
                        tmp = ''  
                        tmpIndex = []    

    # for test
    file_path = os.path.join('./parse_data', rawfile)  # creates file in "parse_data" folder
    with open(file_path, 'w') as f:
        for i in range(len(predOutput_entity)):
            f.write(str(predOutput_entity[i]))
            f.write('\n')
    rd.close()
    
    return predOutput_entity


fRawPath = './input_data/'  

def main():      
    flist = glob.glob('./predict_data/*.txt', recursive = True)  # read file list    
    for file in flist[397:]:
        parse(file)
                              
                        
if __name__ == "__main__":
    main()