# -*- coding: utf-8 -*-
"""
@author: Yaya Liu

Note: Sentence tokenization/counting and word tokenization/counting are based on the rules used by Fable. 

1. sentence tokenization rules

Fable's rule for sentence tokenization:
-> if len(word) > 2 and word[-1] == '.' and (word[:-1].find(word[-1]) == -1)

OR

"&" -> used to identify list


2. word tokenization rule
Fable's rule for word tokenization:
-> space


Known issues:
    1. fable takes punctuations as input. But keeps puctuations will cause cause discrepancies with annotation files. 
       Especially when medications are followed by punctuations: "(", "-", "/"
"""


import glob
import logging
import re


def rawTextPreprocess(file):
    
    '''
    Pre-process medical records  
    Input: raw files of medical records
    Output: processed content of medical records
    
    '''
    
    content = ''
    with open(file, 'r') as f:
        content = f.read()    
        content = re.sub(r'non-insulin', r'non insulin', content) # based on error analysis
        content = re.sub(r'([^\d])/([^\d])', r'\1 \2', content)   # if characters around "/" are not numeric numbers, replace "/" with blank space      
        content = content.replace('(', ' ')
        content = content.replace(')', ' ') 
        content = content.replace('"', ' ')
        content = content.replace('"', ' ')
        content = content.replace(':', ' ')
        content = content.replace('\t-', '& ')  # lists with "-"
        content = content.replace('--', '& ')
        content = content.replace('\n-', '\n&')  
        content = content.replace(' -', '& ')
        content = re.sub(r'&([^\s])', r' \1', content) 
                     
    f.close()
    return content
                
def sentBreak(word):
    '''
    Sentence tokenization check. It follows the fable's rule of Sentence tokenization
    Input: word
    Output: boolean yes or no    
    
    '''
    if (len(word) > 2 and word[-1] == '.' and (word[:-1].find(word[-1]) == -1)) or word == '&':
        return True
    else:
        return False   
    
 
def BIOAnnfile(file):
    '''
    Parse annotation files to extract tokens from medications and create the corresponding BIO labels  
    Input: annatation files
    Output: a nested list. format is [[start of character index, end of character index, Tokens in medications, Bio-lable]]
    
    '''
    BIOlist = []       # store the convert information for medication
       
    with open(file, 'r') as rd:
        for line in rd:
            line = line.strip()             # strip the last '\n' 
            tokens = line.replace(' ', '\t').split('\t')

            if tokens[0][0] == 'T':     # only process medications. The labes start from T
                if ";" in tokens[3]:    # deal with cases like: T2	NoDisposition 2722 2739;2740 2741	Calcium with Vit. D
                    tokens.pop(3)
                
                l = len(tokens)
                
                start_index, end_index = int(tokens[2]), int(tokens[3])                
                
                #print("1111", tokens)
                #print("2222", l, start_index, end_index)
                if l == 5:        # the medication is consist of 1 word
                    #end_index = start_index + len(tokens[4])
                    BIOlist.append([start_index, end_index, tokens[4], 'B-M']) # char index start, char index, medication, Bio-lable
                else:
                    num_words = l - 4        # number of words of the medication's name
                    cnt = 0                  # count the word wihtin a medication
                    while cnt <= num_words - 1:
                        if cnt == 0:
                            end_index = start_index + len(tokens[4])
                            BIOlist.append([start_index, end_index, tokens[4], 'B-M'])
                            cnt += 1
                        else:
                            start_index = start_index + len(tokens[4 + cnt - 1]) + 1
                            end_index = start_index + len(tokens[4 + cnt]) 
                            #print('333333', start_index, end_index)                               
                            BIOlist.append([start_index, end_index, tokens[4 + cnt], 'I-M'])                             
                            cnt += 1                               
        rd.close()
        #print(BIOlist)
        return(BIOlist)

 
def EntityAnnfile(file):
    '''
    Parse annotation files to extract entities(medications) and its start and end index  
    Input: annatation files
    Output: a dictionary
            -- key: start index
            -- value: [end index, entity(medications)]   
    '''
    EntityDict = {}       
       
    with open(file, 'r') as rd:
        for line in rd:
            line = line.strip()             # strip the last '\n' 
            tokens = line.split('\t')
            if tokens[0][0] == 'T':
                tmpIndex = tokens[1].split(' ')
                
                startIndex = tmpIndex[1]
                endIndex = tmpIndex[-1]
                EntityDict[startIndex] = [endIndex, tokens[-1]]
                                       
        rd.close()
        #print(EntityDict)
        return(EntityDict) 
        
   
def rawMappingDict(content):
    '''
    Parse raw medical record files to create a mapping table between word and its location
    Input: raw medical record files 
    
    Output1: dictionary: mappingDict_char
        - key: start of character index
        - value: a list. [end of character index, current word, sentence index, word index]
        
    Output2: dictionary: mappingDict_sent
        - key: (sentence index, word index)
        - value: a list. [start of character index, end of character index, current word]
    
    '''
    
    sentIndex = 1
    wordIndex = 1
    charStart = 0
    #charEnd = 0
    mappingDict_char, mappingDict_sent = {}, {}
    
    
    #with open(file, 'r') as f:
        #content = f.read()  #.split('\n')
        #print(content)
    i = 0
    tmpWord = ''                    
    while i < len(content): 
        if content[i] == ' ' or content[i] == '\t' or content[i] == '\n':
            #print('character is space or \\t or \\n')
            if tmpWord == '':    
                #print('111', charStart)
                i += 1
                charStart += 1
                #print('222', charStart)
            else: 
                #print('333', charStart)
                mappingDict_char[charStart] = [charStart + len(tmpWord), tmpWord, sentIndex, wordIndex]
                mappingDict_sent[(sentIndex, wordIndex)] = [charStart, charStart + len(tmpWord), tmpWord]
                
                i += 1                        
                charStart += len(tmpWord) + 1
                if sentBreak(tmpWord):
                    sentIndex += 1
                    wordIndex = 1
                else:
                    wordIndex += 1
                tmpWord = ''
                #print('444', charStart)
        
        else:
            #print('else', charStart)
            if i != len(content) - 1:   # if it is not the last character
                tmpWord += content[i]
                i += 1   
            else:                    # if it is the last character
                if tmpWord != '':
                    #print('555', charStart)
                    tmpWord += content[i]
                    mappingDict_char[charStart] = [charStart + len(tmpWord), tmpWord, sentIndex, wordIndex] 
                    mappingDict_sent[(sentIndex, wordIndex)] = [charStart, charStart + len(tmpWord), tmpWord]
                    i += 1
                    charStart += len(tmpWord) + 1
                    if sentBreak(tmpWord):
                        sentIndex += 1
                    else:
                        wordIndex += 1
                    tmpWord = ''
                    #print('666', charStart)
    
    #f.close()                      
    #print(mappingDict)
    #logging.info('mapping dict: %s', mappingDict)
    return mappingDict_char, mappingDict_sent


def rawSentlist(file):
    
    '''
    Parse raw medical record files to create a mapping table between sentence index and the content of the sentence
    Input: raw medical record files 
    Output: a dictionary
        - key: sentence index
        - value: the content of the sentence
    
    '''
    
    x_tokens_raw = []
    sentDict = {} 
    sentIndex = 1
    
    with open(file, 'r') as f:
        sents = f.read().split('\n')
        for sent in sents:
            if sent != '':
                x_tokens_raw.extend(sent.split())
    f.close()
        
#    print(x_tokens_raw)
    
    l = 0
    tmpSent = ''
    while l < len(x_tokens_raw):
        if sentBreak(x_tokens_raw[l]):
            #print('111')
            #print(sentIndex, tmpSent)
            if tmpSent == '':
                tmpSent += x_tokens_raw[l]
            else:
                tmpSent += ' ' + x_tokens_raw[l]
            sentDict[sentIndex] = tmpSent
            sentIndex += 1
            tmpSent = ''
        else:     
            #print('222')
            if tmpSent == '':
                tmpSent += x_tokens_raw[l]
            else:
                tmpSent += ' ' + x_tokens_raw[l]
        l += 1
    #print(sentDict)
    return sentDict

 
    
def BIOSentWordIndex(BIOlist, mappingDict_char, file):
    
    '''
    Create a dictionary, key is the token's (sentence index, word index), 
    value is [BIO lable, token from annotation file, token from medical records]
    
    Input: BIO lable list extracted from annotation files, mapping dictionary extracted from raw medical records
    Output: a dictionary
        - key: token's (sentence index, word index)
        - value: [BIO lable, token from annotation file, token from medical records]  
    
    '''    
    BIOSentWordDict = {}  
    tmpTuple = ()
    global countAnn, countMismatched
    
    for i in range(len(BIOlist)):
        countAnn += 1        
        if BIOlist[i][0] in mappingDict_char:    
            value = mappingDict_char[BIOlist[i][0]] 
            #print(BIOlist[i][-2], value[-3])
            
            if BIOlist[i][-2] in value[-3]: #or value[-3] not in BIOlist[i][-2]:
                tmpTuple = (value[-2], value[-1])  # (sentence index, word index)
                if tmpTuple not in BIOSentWordDict.keys():
                    BIOSentWordDict[tmpTuple] = [BIOlist[i][-1], BIOlist[i][-2], value[-3]]
            else:
                #print('Words mismatched.', 'BIO:', BIOlist[i][-2], 'RawText:', value[-3]) 
                logging.info('Words mismatched. BIO: %s ** RawText: %s', BIOlist[i][-2], value[-3])
        else:
            countMismatched += 1           
            #print('Character index mismatched.', BIOlist[i])        
            logging.info('Character index mismatched. %s', BIOlist[i])
            
            with open(file, 'r') as f:
                sents = f.read().split('\n')
                for sent in sents:
                    if BIOlist[i][-2] != '' and BIOlist[i][-2] in sent:
                        #print("### Found ####", sent)
                        logging.info('### Found #### %s', sent) 
                        logging.info("         ")
                f.close()
               
    return  BIOSentWordDict     


log = "utilsLogging.txt"   # create a log file
logging.basicConfig(filename = log, level = logging.DEBUG, format = '%(message)s')  

countAnn = 0
countMismatched = 0      
                
def main():     
    
    fRawlist = glob.glob('./input_data/*.txt', recursive = True)  # read raw medical record file list
    #fRawlist = ["./input_data/399-02.txt"]

    for file in fRawlist[399:]:   
        print("Start Processing...", file)
        logging.info("         ")
        logging.info("         ")
        logging.info("%s Start Processing...", file)
               
        BIOlist = [] 
        mappingDict_char, mappingDict_sent = {}, {}
        BIOSentWordDict = {} 
               
        fileName = file.split('\\')[1].split('.')[0]
        BIOlist = BIOAnnfile('./input_data/' + fileName + '.ann')
        #BIOlist = BIOAnnfile('./input_data/399-02.ann')   
        #print(BIOlist)   
        
        content = rawTextPreprocess(file)
        #print(content)
        
        mappingDict_char, mappingDict_sent  = rawMappingDict(content)
        print(mappingDict_char)
        print()
        print(mappingDict_sent)
        
         
        #BIOSentWordDict = BIOSentWordIndex(BIOlist, mappingDict_char, file)
        
#    print("Total Annotated tokens:", countAnn, "Mismatched tokens:", countMismatched)            
#    print('Mismatched rate:', countMismatched/countAnn)
#    
#    logging.info('Total Annotated tokens: %d ** Mismatched tokens: %d', countAnn, countMismatched)
#    logging.info('Mismatched rate: %f', countMismatched/countAnn)
#    
        
        
if __name__ == "__main__":
    main()
    
