# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 21:11:45 2020

@author: Yaya Liu

1. This script imports the raw text files, and produces sequence labeling for each file based on fable's predictions.

2. The format of the output:
sequence labeling|Token|Sentence ID (starts from zero)

For example:
B-M|Tylenol|0
B-DO|350|0
I-DO|mg|0
B-DU|for|0
I-DU|2|0
I-DU|days|0
...

3. How to run it
- Make sure you can run fable demo - demo.py
- Create file folders "input_data" and "predict_data" under the same directory with fable model
- Put raw text files under the folder: "input_data" 
- Run this file under Linux system. Output will be generated under folder "predict_data"

"""

import fable
import os, glob

def main():    
    flist = glob.glob('./input_data/*.txt', recursive = True)  # read file list
    for file in flist:        
        tokens, labels, ids = fable.fable(file)  # return 3 lists: tokens, prediction labels, sentence ID
                
        file_name = file[-10:]             # intend to use split function, but it did not work in linux
        file_path = os.path.join('./predict_data', file_name)  # create file in "predict_data" folder
        
        with open(file_path, 'w') as f:
            for i in range(len(tokens)):
                #if labels[i] != 'NO':
                tmp = labels[i] + '|' + tokens[i] + '|' + str(ids[i]) + '\n'  # reconstruct output string as "label|token|sentenceID"
                f.write(tmp)
        f.close()

if __name__ == "__main__":
    main()
    
    