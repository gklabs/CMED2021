# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 22:02:57 2020

@author: Yaya Liu

1. This scripts interprets outputs(sequence labeling) from "predict.py", and maps the sequence labeling to Drug/Dosage...

2. The format of the output:
NER label|tokens|Sentence ID (starts from zero)
    
For example:
Drug|heparin|6
Drug|ASA|6
Drug|amio.|10
Drug|amio|15
Route|drip,|15
Drug|BB,|15
Drug|plavix, ASA, lipitor,|15
Drug|integrilin|15
Drug|Avandia|15
Drug|ASA, Lipitor|17
Dosage|20,|17
    
3. How to run it
- Make sure you can run predict.py
- Create file folder "parse_data" under the same directory with fable model
- Run this file under Linux system/Windows. Output will be generated under folder "parse_data"


"""

import glob
import os

# dictionary for label transformation
lookup = {        
            'M': 'Drug',
            'DO': 'Dosage',   
            'F': 'Frequency',
            'MO': 'Route',
            'R': 'Reason',                        
            'DU': 'Duration'        
        }


def main():
    
    flist = glob.glob('./predict_data/*.txt', recursive = True)  # read file list
    
    for file in flist:
        out = []
        
        with open(file, 'r') as rd:
            for line in rd:
                line = line.strip()      # strip the last '\n'                
                label = line.split('|')[0]
                if label != 'NO': # and label.split('-')[1] == 'M': 
                    out.append(line)    # only save outputs that has NER labeling
        #print(out)
        
        s_list = []
        tmp = '' 
        for i in range(len(out)):
            #prev_position = out[i].split('|')[0].split('-')[0]   # label position
            prev_concept = out[i].split('|')[0].split('-')[1]    # label tab            
            prev_value = out[i].split('|')[1]                    # value
            sent_id = out[i].split('|')[2]
            tmp += prev_value + ' '

            #print("previous: ", prev_position, prev_concept, prev_value)
            #print("previous_tmp: ", tmp)
            
            if i == len(out) - 1:
                s = lookup[prev_concept] + '|' + tmp.rstrip() + '|' + sent_id
                s_list.append(s)
                tmp = ''
            else:
                next_position = out[i + 1].split('|')[0].split('-')[0]   # label position
                if next_position == 'B':  
                    s = lookup[prev_concept] + '|' + tmp.rstrip() + '|' + sent_id
                    s_list.append(s)
                    tmp = ''
                
        #print(s_list)
        file_name = file[-10:]             # intend to use split function, but it did not work in linux
        file_path = os.path.join('./parse_data', file_name)  # creates file in "parse_data" folder
        with open(file_path, 'w') as f:
            for i in range(len(s_list)):
                f.write(s_list[i] + '\n')
        rd.close()

if __name__ == "__main__":
    main()