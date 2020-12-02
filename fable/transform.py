# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 21:21:26 2020

@author: Yaya Liu

1. 

Objectives
- Filter out the drugs (finished)
- Map sentence ID to the postions of the start and end tokens in the sentence. (working...)
- Locate mediation's position


"""

import glob
import os

def main():
    flist = glob.glob('./parse_data/*.txt', recursive = True)  # read file list
    
    for file in flist:
        out = []
        with open(file, 'r') as rd:
            for line in rd:
                line = line.strip()      # strip the last '\n'
                label = line.split('|')[0]
                if label == 'Drug': 
                    out.append(line)    # only save result that has drug labels

        file_name = file[-10:]             # intend to use split function, but it did not work in linux
        file_path = os.path.join('./transform_data', file_name)  # creates file in "parse_data" folder
        with open(file_path, 'w') as f:
            for i in range(len(out)):
                f.write(out[i] + '\n')
        rd.close()




if __name__ == '__main__':
    main()