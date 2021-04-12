# Fable. By Carson Tao

import pickle, re, os
from nltk import pos_tag
from fablelib import *
import utils   # Yaya Liu add 


single_temporal_signal = ['D', 'D-day', 'Dd']
single_temporal = ['day', 'days', 'course', 'week', 'weeks', 'months', 'month', 'overnight']
a_period_phrase = ['course', 'days']
a_during_term = ['admission', 'course', 'hospitalization', 'stay']
a_for_term = ['days', 'course', 'week', 'weeks', 'months', 'month', 'overnight', 
              'hours', 'period', 'doses', 'stay', 'years', 'yrs']
throughout_term = ['stay', 'hospitalization', 'admission']
x_terms = ['dose', 'days', 'doses', 'day']
xD_terms = ['day', 'days', 'doses', 'course', 'prn', 'p.r.n', 'weeks']

def fable_demo(fname):
    '''Extracts some drugs'''
    try:
        with open(fname) as f:
            x_tokens_raw = []
            sents = f.read().split('\n')
            for sent in sents:
                if sent != '':
                    x_tokens_raw.extend(sent.split())
            f.close()

        x_tokens_raw.extend(['|', '|'])
        
        x_tokens = []
        for tok in x_tokens_raw:
            x_tokens.append(normalise_token(tok))
            
        #print(x_tokens)
        
        pos_raw = pos_tag(x_tokens_raw)
        #print(pos_raw)
        
        pos_tags = []
        for tag in pos_raw:
            pos_tags.append(tag[1])
            
            
        crf_label = ['O'] * len(x_tokens)
        x_vec = []
        for tok in x_tokens:
            x_vec.append(word2vec_ih(tok))
            
        nul = word2vec_ih('4')
        #print(nul)
        unk = word2vec_ih('<unk>')
        #print(unk)
        
        for pos, vec in enumerate(x_vec):
            if vec == nul:
                x_vec[pos] = unk
                
        #print(len(x_tokens_raw), len(x_tokens), len(x_vec))
                
        x_duration = []
        x_duration = ['not_du']*len(x_tokens)
        # Basic Rules - Mark III
        # Single_Temporal_Signal
        co = 0
        while co < len(x_tokens):
            pos = ''
            if x_tokens[co] in single_temporal_signal and x_tokens[co+1] in single_temporal:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                pos = co+1
            elif x_tokens[co] in single_temporal_signal and x_tokens[co+2] in single_temporal:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                pos = co+2
            else:
                pos = co
            co = pos + 1
        # A_Period_Phrase
        co = 0
        while co < len(x_tokens):
            pos = ''
            if x_tokens[co] == 'a' and x_tokens[co+1] in a_period_phrase:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                pos = co+1
            elif x_tokens[co] == 'a' and x_tokens[co+2] in a_period_phrase:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                pos = co+2
            elif x_tokens[co] == 'a' and x_tokens[co+3] in a_period_phrase:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                x_duration[co+3] = 'duration'
                pos = co+3
            elif x_tokens[co] == 'a' and x_tokens[co+4] in a_period_phrase:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                x_duration[co+3] = 'duration'
                x_duration[co+4] = 'duration'
                pos = co+4
            else:
                pos = co
            co = pos + 1
        # A_During_Term
        co = 0
        while co < len(x_tokens):
            pos = ''
            if x_tokens[co] == 'during' and x_tokens[co+1] in a_during_term:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                pos = co+1
            elif x_tokens[co] == 'during' and x_tokens[co+2] in a_during_term:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                pos = co+2
            elif x_tokens[co] == 'during' and x_tokens[co+3] in a_during_term:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                x_duration[co+3] = 'duration'
                pos = co+3
            elif x_tokens[co] == 'during' and x_tokens[co+4] in a_during_term:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                x_duration[co+3] = 'duration'
                x_duration[co+4] = 'duration'
                pos = co+4
            elif x_tokens[co] == 'during' and x_tokens[co+5] in a_during_term:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                x_duration[co+3] = 'duration'
                x_duration[co+4] = 'duration'
                x_duration[co+5] = 'duration'
                pos = co+5
            else:
                pos = co
            co = pos + 1
        # A_For_Term
        co = 0
        while co < len(x_tokens):
            pos = ''
            if x_tokens[co] == 'for' and x_tokens[co+1] in a_for_term:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                pos = co+1
            elif x_tokens[co] == 'for' and x_tokens[co+2] in a_for_term:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                pos = co+2
            elif x_tokens[co] == 'for' and x_tokens[co+3] in a_for_term:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                x_duration[co+3] = 'duration'
                pos = co+3
            elif x_tokens[co] == 'for' and x_tokens[co+4] in a_for_term:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                x_duration[co+3] = 'duration'
                x_duration[co+4] = 'duration'
                pos = co+4
            elif x_tokens[co] == 'for' and x_tokens[co+5] in a_for_term:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                x_duration[co+3] = 'duration'
                x_duration[co+4] = 'duration'
                x_duration[co+5] = 'duration'
                pos = co+5
            elif x_tokens[co] == 'for' and x_tokens[co+6] in a_for_term:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                x_duration[co+3] = 'duration'
                x_duration[co+4] = 'duration'
                x_duration[co+5] = 'duration'
                x_duration[co+6] = 'duration'
                pos = co+6
            else:
                pos = co
            co = pos + 1
        # Number_of_doses
        co = 0
        while co < len(x_tokens):
            pos = ''
            if x_tokens[co] == 'number' and x_tokens[co+1] == 'of' and x_tokens[co+2] == 'doses' and x_tokens[co+3] == 'required':
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                x_duration[co+3] = 'duration'
                x_duration[co+4] = 'duration'
                x_duration[co+5] = 'duration'
                x_duration[co+6] = 'duration'
                x_duration[co+7] = 'duration'
                pos = co+7
            else:
                pos = co
            co = pos + 1
        # throughout  sometime
        co = 0
        while co < len(x_tokens):
            pos = ''
            if x_tokens[co] == 'throughout' and x_tokens[co+1] in throughout_term:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                pos = co+1
            elif x_tokens[co] == 'throughout' and x_tokens[co+2] in throughout_term:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                pos = co+2
            elif x_tokens[co] == 'throughout' and x_tokens[co+3] in throughout_term:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                x_duration[co+3] = 'duration'
                pos = co+3
            else:
                pos = co
            co = pos + 1
        # times
        co = 0
        while co < len(x_tokens):
            pos = ''
            if x_tokens[co] == 'times' and x_tokens[co+1] == 'D' and x_tokens[co+2] == 'days':
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                pos = co+2
            else:
                pos = co
            co = pos + 1
        # Until
        co = 0
        while co < len(x_tokens):
            pos = ''
            if x_tokens[co] == 'until' and x_tokens[co+1] == 'D/D/D':
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                pos = co+1
            elif x_tokens[co] == 'until' and x_tokens[co+1] == 'D/D':
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                pos = co+1
            elif x_tokens[co] == 'until' and x_tokens[co+2] == 'D/D/D':
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                pos = co+2
            else:
                pos = co
            co = pos + 1
        # x multiplication
        co = 0
        while co < len(x_tokens):
            pos = ''
            if x_tokens[co] == 'x' and x_tokens[co+1] == 'D':
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                pos = co+1
            elif x_tokens[co] == 'x' and x_tokens[co+1] == 'D' and x_tokens[co+2] in x_terms:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                pos = co+2
            else:
                pos = co
            co = pos + 1
        # xD multiplication
        co = 0
        while co < len(x_tokens):
            pos = ''
            if x_tokens[co] == 'xD' and x_tokens[co+1] in xD_terms:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                pos = co+1
            elif x_tokens[co] == 'x' and x_tokens[co+2] in xD_terms:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                pos = co+2
            else:
                pos = co
            co = pos + 1
        crf_doc = open('data/temp', 'w')
        symbols = '.'
        co = 0
        while co < len(x_tokens) - 2:
            we = vec2crfsuite(co, x_vec)
            #print(co)
            if len(x_tokens_raw[co]) > 2 and x_tokens_raw[co][-1] == '.' and (x_tokens_raw[co][:-1].find(x_tokens_raw[co][-1]) == -1):
                #print('111', x_tokens_raw[co], len(x_tokens_raw[co]))
                crf_doc.write(crf_label[co]+'\tw[-2]='+x_tokens[co-2]+'\tw[-1]='+x_tokens[co-1]+'\tw[0]='+x_tokens[co]+'\tw[+1]='+x_tokens[co+1]+'\tw[+2]='+x_tokens[co+2]+'\tpos[-2]='
                             +pos_tags[co-2]+'\tpos[-1]='+pos_tags[co-1]+'\tpos[0]='+pos_tags[co]+'\tpos[+1]='+pos_tags[co+1]+'\tpos[+2]='+pos_tags[co+2]+'\tduration[-2]='
                             +x_duration[co-2]+'\tduration[-1]='+x_duration[co-1]+'\tduration[0]='+x_duration[co]+'\tduration[+1]='+x_duration[co+1]+'\tduration[+2]='+x_duration[co+2]+'\t'+we+'\n')
                crf_doc.write('\n')
            else:
                #print('222', x_tokens_raw[co], len(x_tokens_raw[co]))
                crf_doc.write(crf_label[co]+'\tw[-2]='+x_tokens[co-2]+'\tw[-1]='+x_tokens[co-1]+'\tw[0]='+x_tokens[co]+'\tw[+1]='+x_tokens[co+1]+'\tw[+2]='+x_tokens[co+2]+'\tpos[-2]='
                             +pos_tags[co-2]+'\tpos[-1]='+pos_tags[co-1]+'\tpos[0]='+pos_tags[co]+'\tpos[+1]='+pos_tags[co+1]+'\tpos[+2]='+pos_tags[co+2]+'\tduration[-2]='
                             +x_duration[co-2]+'\tduration[-1]='+x_duration[co-1]+'\tduration[0]='+x_duration[co]+'\tduration[+1]='+x_duration[co+1]+'\tduration[+2]='+x_duration[co+2]+'\t'+we+'\n')
            co+=1
        crf_doc.close()
        os.system('crfsuite/bin/crfsuite tag -m data/fable.model -r data/temp > predicted')
        predictions = []
        with open('predicted') as f:
            results = f.read().split('\n')
            for result in results:
                if result != '':
                    predictions.append(result.split('\t')[1])
            f.close()
        print('Processed %s tokens.' % str(len(predictions)))
        y_labels = []
        for pos, label in enumerate(predictions):
            if label == 'O':
                y_labels.append('NO')
            else:
                y_labels.append(label)
        #print(y_labels)
        for pos, label in enumerate(y_labels):
        	if len(label) == 1:
	            print(label + '     ' + x_tokens_raw[pos])
	        else:
                    print(label + '     ' + x_tokens_raw[pos])
    except IOError:
        print('File NOT found.')



'''
Yaya Liu add
fname: file path + file name
flag = 0: perform prediction (NER)
flag = 1: prepare training files to create new fable model
'''

def fable(fname, flag, currFold):
    '''Extracts some drugs'''
    try:
#        with open(fname) as f:
#            x_tokens_raw = []
#            sents = f.read().split('\n')
#            for sent in sents:
#                if sent != '':
#                    x_tokens_raw.extend(sent.split())
#            f.close()
#            
        content = utils.rawTextPreprocess(fname)  # Yaya Liu add       
        x_tokens_raw = []
        sents = content.split('\n')
        for sent in sents:
            if sent != '':
                x_tokens_raw.extend(sent.split())    
                
        x_tokens_raw.extend(['|', '|'])       
        
        x_tokens = []
        for tok in x_tokens_raw:
            x_tokens.append(normalise_token(tok))
        
        pos_raw = pos_tag(x_tokens_raw)
        pos_tags = []
        for tag in pos_raw:
            pos_tags.append(tag[1])
        
        crf_label = ['O'] * len(x_tokens)
        x_vec = []
        for tok in x_tokens:
            x_vec.append(word2vec_ih(tok))
    
        nul = word2vec_ih('4')
        unk = word2vec_ih('<unk>')
        for pos, vec in enumerate(x_vec):
            if vec == nul:
                x_vec[pos] = unk
        x_duration = []
        x_duration = ['not_du']*len(x_tokens)
        # Basic Rules - Mark III
        # Single_Temporal_Signal
        co = 0
        while co < len(x_tokens):
            pos = ''
            if x_tokens[co] in single_temporal_signal and x_tokens[co+1] in single_temporal:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                pos = co+1
            elif x_tokens[co] in single_temporal_signal and x_tokens[co+2] in single_temporal:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                pos = co+2
            else:
                pos = co
            co = pos + 1
        # A_Period_Phrase
        co = 0
        while co < len(x_tokens):
            pos = ''
            if x_tokens[co] == 'a' and x_tokens[co+1] in a_period_phrase:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                pos = co+1
            elif x_tokens[co] == 'a' and x_tokens[co+2] in a_period_phrase:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                pos = co+2
            elif x_tokens[co] == 'a' and x_tokens[co+3] in a_period_phrase:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                x_duration[co+3] = 'duration'
                pos = co+3
            elif x_tokens[co] == 'a' and x_tokens[co+4] in a_period_phrase:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                x_duration[co+3] = 'duration'
                x_duration[co+4] = 'duration'
                pos = co+4
            else:
                pos = co
            co = pos + 1
        # A_During_Term
        co = 0
        while co < len(x_tokens):
            pos = ''
            if x_tokens[co] == 'during' and x_tokens[co+1] in a_during_term:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                pos = co+1
            elif x_tokens[co] == 'during' and x_tokens[co+2] in a_during_term:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                pos = co+2
            elif x_tokens[co] == 'during' and x_tokens[co+3] in a_during_term:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                x_duration[co+3] = 'duration'
                pos = co+3
            elif x_tokens[co] == 'during' and x_tokens[co+4] in a_during_term:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                x_duration[co+3] = 'duration'
                x_duration[co+4] = 'duration'
                pos = co+4
            elif x_tokens[co] == 'during' and x_tokens[co+5] in a_during_term:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                x_duration[co+3] = 'duration'
                x_duration[co+4] = 'duration'
                x_duration[co+5] = 'duration'
                pos = co+5
            else:
                pos = co
            co = pos + 1
        # A_For_Term
        co = 0
        while co < len(x_tokens):
            #print(x_tokens[co])  
            pos = ''
            if x_tokens[co] == 'for' and co + 1 < len(x_tokens) and x_tokens[co+1] in a_for_term:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                pos = co+1
            elif x_tokens[co] == 'for' and co + 2 < len(x_tokens) and x_tokens[co+2] in a_for_term:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                pos = co+2
            elif x_tokens[co] == 'for' and co + 3 < len(x_tokens) and x_tokens[co+3] in a_for_term:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                x_duration[co+3] = 'duration'
                pos = co+3
            elif x_tokens[co] == 'for' and co + 4 < len(x_tokens) and x_tokens[co+4] in a_for_term:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                x_duration[co+3] = 'duration'
                x_duration[co+4] = 'duration'
                pos = co+4
            elif x_tokens[co] == 'for' and co + 5 < len(x_tokens) and x_tokens[co+5] in a_for_term:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                x_duration[co+3] = 'duration'
                x_duration[co+4] = 'duration'
                x_duration[co+5] = 'duration'
                pos = co+5
            elif x_tokens[co] == 'for' and co + 6 < len(x_tokens) and x_tokens[co+6] in a_for_term:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                x_duration[co+3] = 'duration'
                x_duration[co+4] = 'duration'
                x_duration[co+5] = 'duration'
                x_duration[co+6] = 'duration'
                pos = co+6
            else:
                pos = co
            co = pos + 1
        # Number_of_doses
        co = 0
        while co < len(x_tokens):
            pos = ''
            if x_tokens[co] == 'number' and x_tokens[co+1] == 'of' and x_tokens[co+2] == 'doses' and x_tokens[co+3] == 'required':
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                x_duration[co+3] = 'duration'
                x_duration[co+4] = 'duration'
                x_duration[co+5] = 'duration'
                x_duration[co+6] = 'duration'
                x_duration[co+7] = 'duration'
                pos = co+7
            else:
                pos = co
            co = pos + 1
        # throughout  sometime
        co = 0
        while co < len(x_tokens):
            pos = ''
            if x_tokens[co] == 'throughout' and x_tokens[co+1] in throughout_term:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                pos = co+1
            elif x_tokens[co] == 'throughout' and x_tokens[co+2] in throughout_term:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                pos = co+2
            elif x_tokens[co] == 'throughout' and x_tokens[co+3] in throughout_term:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                x_duration[co+3] = 'duration'
                pos = co+3
            else:
                pos = co
            co = pos + 1
        # times
        co = 0
        while co < len(x_tokens):
            pos = ''
            if x_tokens[co] == 'times' and x_tokens[co+1] == 'D' and x_tokens[co+2] == 'days':
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                pos = co+2
            else:
                pos = co
            co = pos + 1
        # Until
        co = 0
        while co < len(x_tokens):
            pos = ''
            if x_tokens[co] == 'until' and x_tokens[co+1] == 'D/D/D':
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                pos = co+1
            elif x_tokens[co] == 'until' and x_tokens[co+1] == 'D/D':
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                pos = co+1
            elif x_tokens[co] == 'until' and x_tokens[co+2] == 'D/D/D':
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                pos = co+2
            else:
                pos = co
            co = pos + 1
        # x multiplication
        co = 0
        while co < len(x_tokens):
            pos = ''
            if x_tokens[co] == 'x' and x_tokens[co+1] == 'D':
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                pos = co+1
            elif x_tokens[co] == 'x' and x_tokens[co+1] == 'D' and x_tokens[co+2] in x_terms:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                pos = co+2
            else:
                pos = co
            co = pos + 1
        # xD multiplication
        co = 0
        while co < len(x_tokens):
            pos = ''
            if x_tokens[co] == 'xD' and x_tokens[co+1] in xD_terms:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                pos = co+1
            elif x_tokens[co] == 'x' and x_tokens[co+2] in xD_terms:
                x_duration[co] = 'duration'
                x_duration[co+1] = 'duration'
                x_duration[co+2] = 'duration'
                pos = co+2
            else:
                pos = co
            co = pos + 1
            
        symbols = '.'        
        co = 0
            
        if flag == 0:  # perform prediction
            crf_doc = open('data/temp', 'w')                  
            while co < len(x_tokens) - 2:
                we = vec2crfsuite(co, x_vec)            
                if x_tokens_raw[co] == '&':          # Yaya Liu add
                    crf_doc.write(crf_label[co]+'\tw[-2]='+x_tokens[co-2]+'\tw[-1]='+x_tokens[co-1]+'\tw[0]='+x_tokens[co]+'\tw[+1]='+x_tokens[co+1]+'\tw[+2]='+x_tokens[co+2]+'\tpos[-2]='
                     +pos_tags[co-2]+'\tpos[-1]='+pos_tags[co-1]+'\tpos[0]='+pos_tags[co]+'\tpos[+1]='+pos_tags[co+1]+'\tpos[+2]='+pos_tags[co+2]+'\tduration[-2]='
                     +x_duration[co-2]+'\tduration[-1]='+x_duration[co-1]+'\tduration[0]='+x_duration[co]+'\tduration[+1]='+x_duration[co+1]+'\tduration[+2]='+x_duration[co+2]+'\t'+we+'\n')
                    crf_doc.write('\n')
                     
                elif len(x_tokens_raw[co]) > 2 and x_tokens_raw[co][-1] == '.' and (x_tokens_raw[co][:-1].find(x_tokens_raw[co][-1]) == -1):
                    crf_doc.write(crf_label[co]+'\tw[-2]='+x_tokens[co-2]+'\tw[-1]='+x_tokens[co-1]+'\tw[0]='+x_tokens[co]+'\tw[+1]='+x_tokens[co+1]+'\tw[+2]='+x_tokens[co+2]+'\tpos[-2]='
                                 +pos_tags[co-2]+'\tpos[-1]='+pos_tags[co-1]+'\tpos[0]='+pos_tags[co]+'\tpos[+1]='+pos_tags[co+1]+'\tpos[+2]='+pos_tags[co+2]+'\tduration[-2]='
                                 +x_duration[co-2]+'\tduration[-1]='+x_duration[co-1]+'\tduration[0]='+x_duration[co]+'\tduration[+1]='+x_duration[co+1]+'\tduration[+2]='+x_duration[co+2]+'\t'+we+'\n')
                    crf_doc.write('\n')
                
                else:
                    crf_doc.write(crf_label[co]+'\tw[-2]='+x_tokens[co-2]+'\tw[-1]='+x_tokens[co-1]+'\tw[0]='+x_tokens[co]+'\tw[+1]='+x_tokens[co+1]+'\tw[+2]='+x_tokens[co+2]+'\tpos[-2]='
                                 +pos_tags[co-2]+'\tpos[-1]='+pos_tags[co-1]+'\tpos[0]='+pos_tags[co]+'\tpos[+1]='+pos_tags[co+1]+'\tpos[+2]='+pos_tags[co+2]+'\tduration[-2]='
                                 +x_duration[co-2]+'\tduration[-1]='+x_duration[co-1]+'\tduration[0]='+x_duration[co]+'\tduration[+1]='+x_duration[co+1]+'\tduration[+2]='+x_duration[co+2]+'\t'+we+'\n')

                co+=1
                
            crf_doc.close()
            
            if currFold == 0:
                os.system('crfsuite/bin/crfsuite tag -m data/newModel_0.model -r data/temp > predicted')
            elif currFold == 1:
                os.system('crfsuite/bin/crfsuite tag -m data/newModel_1.model -r data/temp > predicted')
            elif currFold == 2:
                os.system('crfsuite/bin/crfsuite tag -m data/newModel_2.model -r data/temp > predicted')
            elif currFold == 3:
                os.system('crfsuite/bin/crfsuite tag -m data/newModel_3.model -r data/temp > predicted')
            elif currFold == -1:
                os.system('crfsuite/bin/crfsuite tag -m data/fable.model -r data/temp > predicted')
                            
            predictions = []
            
            sent_index = []             #Yaya Liu added 2020/11/09, store sentence index, start from 1
            word_index = []             #Yaya Liu added, store word index in a sentence, start from 1
            
            with open('predicted') as f:            
                sent_count = 1               #Yaya Liu added 2020/11/09, count sentence      
                word_count = 1               #Yaya Liu added, count word 
                results = f.read().split('\n')            
                for result in results:
                    if result != '':
                        predictions.append(result.split('\t')[1])
                        sent_index.append(sent_count)
                        word_index.append(word_count)
                        word_count += 1
                    else:
                        sent_count += 1
                        word_count = 1
                f.close()
            
            y_labels = []
            for pos, label in enumerate(predictions):
                if label == 'O':
                    y_labels.append('NO')
                else:
                    y_labels.append(label)
            return x_tokens_raw[:-2], y_labels, sent_index, word_index
        
        elif flag == 1:         # Yaya Liu add prepare training files
            fileName = fname.split('/')[2].split('.')[0]           
            filePath = './trainning_build_data/' + fileName + '.train'    
            training_doc = open(filePath, 'w')                    
            
            s_count = 1               #record sentence index    
            w_count = 1               #record word index  
            
            while co < len(x_tokens) - 2:
                we = vec2crfsuite(co, x_vec)            
                if x_tokens_raw[co] == '&':          # Yaya Liu add                 
                    training_doc.write(crf_label[co]+'\tw[-2]='+x_tokens[co-2]+'\tw[-1]='+x_tokens[co-1]+'\tw[0]='+x_tokens[co]+'\tw[+1]='+x_tokens[co+1]+'\tw[+2]='+x_tokens[co+2]+'\tpos[-2]='
                     +pos_tags[co-2]+'\tpos[-1]='+pos_tags[co-1]+'\tpos[0]='+pos_tags[co]+'\tpos[+1]='+pos_tags[co+1]+'\tpos[+2]='+pos_tags[co+2]+'\tduration[-2]='
                     +x_duration[co-2]+'\tduration[-1]='+x_duration[co-1]+'\tduration[0]='+x_duration[co]+'\tduration[+1]='+x_duration[co+1]+'\tduration[+2]='+x_duration[co+2]+'\t'+we+'\t'+str(s_count)+'\t'+str(w_count)+'\n')
                    training_doc.write('\n')   
                    
                    s_count += 1
                    w_count = 1
                     
                elif len(x_tokens_raw[co]) > 2 and x_tokens_raw[co][-1] == '.' and (x_tokens_raw[co][:-1].find(x_tokens_raw[co][-1]) == -1):                    
                    training_doc.write(crf_label[co]+'\tw[-2]='+x_tokens[co-2]+'\tw[-1]='+x_tokens[co-1]+'\tw[0]='+x_tokens[co]+'\tw[+1]='+x_tokens[co+1]+'\tw[+2]='+x_tokens[co+2]+'\tpos[-2]='
                                 +pos_tags[co-2]+'\tpos[-1]='+pos_tags[co-1]+'\tpos[0]='+pos_tags[co]+'\tpos[+1]='+pos_tags[co+1]+'\tpos[+2]='+pos_tags[co+2]+'\tduration[-2]='
                                 +x_duration[co-2]+'\tduration[-1]='+x_duration[co-1]+'\tduration[0]='+x_duration[co]+'\tduration[+1]='+x_duration[co+1]+'\tduration[+2]='+x_duration[co+2]+'\t'+we+'\t'+str(s_count)+'\t'+str(w_count)+'\n')
                    training_doc.write('\n')
                    
                    s_count += 1
                    w_count = 1
                
                else:    
                    training_doc.write(crf_label[co]+'\tw[-2]='+x_tokens[co-2]+'\tw[-1]='+x_tokens[co-1]+'\tw[0]='+x_tokens[co]+'\tw[+1]='+x_tokens[co+1]+'\tw[+2]='+x_tokens[co+2]+'\tpos[-2]='
                                 +pos_tags[co-2]+'\tpos[-1]='+pos_tags[co-1]+'\tpos[0]='+pos_tags[co]+'\tpos[+1]='+pos_tags[co+1]+'\tpos[+2]='+pos_tags[co+2]+'\tduration[-2]='
                                 +x_duration[co-2]+'\tduration[-1]='+x_duration[co-1]+'\tduration[0]='+x_duration[co]+'\tduration[+1]='+x_duration[co+1]+'\tduration[+2]='+x_duration[co+2]+'\t'+we+'\t'+str(s_count)+'\t'+str(w_count)+'\n') 
                    w_count += 1
                co+=1
            training_doc.close()           
                
    except IOError:
        print('File NOT found.')
