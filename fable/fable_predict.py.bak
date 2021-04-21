# -*- coding: utf-8 -*-
"""
@author: Yaya Liu

1. This script imports the raw text files, and call fable to predict sequence labeling for each files

2. How to run it
- Make sure you can run fable demo - demo.py under a Linux system or Linux-like system
- Create file folders "input_data" and "predict_data" under the same directory with fable model
- Put raw text files under the folder: "input_data" 
- Run this file under a Linux system or Linux-like system. Output will be generated under folder "predict_data"

"""

import fable
import os, glob


"""
Input:
    - file: raw text files
    - flag: if value is 0, fable will perform prediction. If value is 1, fable will prepare training files
    - currFold: current fold of K-fold cross validation
    
Output: files with predictions on tokens, will be saved under folder "predict_data"

- format
BIO label|Word|Sentence index|Word index

For example:
B-M|Tylenol|1|1
B-DO|350|1|2
I-DO|mg|1|3
B-DU|for|1|4
I-DU|2|1|5
I-DU|days|1|6

"""
def predict(file, flag, currFold):     
    print("predicting...", file)
    tokens, labels, sent_index, word_index = fable.fable(file, flag, currFold)  # return 4 lists: tokens, prediction labels, sentence index, word index
            
    #file_name = file[-10:]             # intend to use split function, but it did not work in linux        
    file_name = file.split('/')[2]
    file_path = os.path.join('./predict_data', file_name)  # create file in "predict_data" folder
    
    with open(file_path, 'w') as f:
        for i in range(len(tokens)):
            #if labels[i] != 'NO':
            tmp = labels[i] + '|' + tokens[i] + '|' + str(sent_index[i]) +  '|' + str(word_index[i]) + '\n'  # reconstruct output string as "label|token|sentenceID"
            f.write(tmp)
    f.close()

def main(): 
    flist = glob.glob('./input_data/*.txt', recursive = True)  # read file list
    for file in flist[300:]:
        predict(file, 0, 0)
        
if __name__ == "__main__":
    main()
    
    