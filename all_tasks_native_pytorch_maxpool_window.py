'''
all_tasks_sliding_window.py
'''


import torch
import pandas as  pd
import sys
from sklearn import preprocessing
import torch
import os
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import AutoTokenizer,AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments, AdamW, AutoConfig
from sklearn.metrics import classification_report,confusion_matrix
from collections import defaultdict
from tqdm import tqdm
import spacy
import pickle


ids_mistake =[]

# class shared_learner_base(nn.Module):
#   def __init__(self,bert,output_dims):
#     super().__init__() #call super class constructor for nn.module
          
#     self.bert = bert
#     self.output_dim_disposition = output_dims['disposition']
#     self.output_dim_actor = output_dims['actor']
#     self.output_dim_certainty = output_dims['certainty']
#     self.output_dim_temporality =output_dims['temporality']
#     self.output_dim_action = output_dims['action']
#     self.loss = nn.CrossEntropyLoss()

#     self.embedding_dim = 1536
#     #go into BERT's embedding func
#     self.softmax = nn.Softmax()

#     #direct output layer from BERT
#     self.op_disposition = nn.Linear(self.embedding_dim,self.output_dim_disposition)
#     self.op_actor = nn.Linear(self.embedding_dim, self.output_dim_actor)
#     self.op_certainty = nn.Linear(self.embedding_dim, self.output_dim_certainty)
#     self.op_temporality = nn.Linear(self.embedding_dim, self.output_dim_temporality)
#     self.op_action = nn.Linear(self.embedding_dim, self.output_dim_action)

      
#   def forward(self,input_ids,attention_mask,med_mask,labels):
#     # all_losses = {}

#     bert_op = self.bert(input_ids,attention_mask=attention_mask)
#     pooled_output = bert_op['pooler_output'] #sentence embedding from BERT
#     h= bert_op['last_hidden_state']

#     h_mask = med_mask.unsqueeze(2).repeat(1, 1, h.shape[2])
#     med_embedding = (h*h_mask).max(dim=1)[0]
#     features = torch.cat([pooled_output,med_embedding ], dim=-1)


#     #output from  
#     disposition = self.op_disposition(features) 
#     certainty = self.op_certainty(features)
#     temporality = self.op_temporality(features)
#     action= self.op_action(features)
#     actor= self.op_actor(features)
        
#     outputs = (disposition,certainty,temporality,action,actor)
    
#     return outputs

class shared_learner_base(nn.Module):
  def __init__(self,bert,output_dims):
    super().__init__() #call super class constructor for nn.module
          
    self.bert = bert
    self.output_dim_disposition = output_dims['disposition']
    self.output_dim_actor = output_dims['actor']
    self.output_dim_certainty = output_dims['certainty']
    self.output_dim_temporality =output_dims['temporality']
    self.output_dim_action = output_dims['action']
    self.loss = nn.CrossEntropyLoss()

    self.embedding_dim = 768
    #go into BERT's embedding func
    self.softmax = nn.Softmax()

    # #direct output layer from BERT
    # self.op_disposition = nn.Linear(self.embedding_dim,self.output_dim_disposition)
    # self.op_actor = nn.Linear(self.embedding_dim, self.output_dim_actor)
    # self.op_certainty = nn.Linear(self.embedding_dim, self.output_dim_certainty)
    # self.op_temporality = nn.Linear(self.embedding_dim, self.output_dim_temporality)
    # self.op_action = nn.Linear(self.embedding_dim, self.output_dim_action)

    #hierarchical multi-task
    self.op_disposition = nn.Linear(self.embedding_dim,self.output_dim_disposition)
    self.op_actor = nn.Linear(self.embedding_dim + self.output_dim_disposition, self.output_dim_actor)
    self.op_certainty = nn.Linear(self.embedding_dim + self.output_dim_disposition, self.output_dim_certainty)
    self.op_temporality = nn.Linear(self.embedding_dim + self.output_dim_disposition, self.output_dim_temporality)
    self.op_action = nn.Linear(self.embedding_dim + self.output_dim_disposition, self.output_dim_action)

      
  def forward(self,input_ids,attention_mask,med_mask,labels):
    # all_losses = {}

    bert_op = self.bert(input_ids,attention_mask=attention_mask)
    pooled_output = bert_op['pooler_output'] #sentence embedding from BERT
    # h= bert_op['last_hidden_state']

    # h_mask = med_mask.unsqueeze(2).repeat(1, 1, h.shape[2])
    # med_embedding = (h*h_mask).max(dim=1)[0]
    # features = torch.cat([pooled_output,med_embedding ], dim=-1)

    
    # #plain multi-task  
    # features = pooled_output
    # disposition = self.op_disposition(features) 
    # certainty = self.op_certainty(features)
    # temporality = self.op_temporality(features)
    # action= self.op_action(features)
    # actor= self.op_actor(features)

    # hierarchical multi-task
    disposition = self.op_disposition(pooled_output) 
    features = torch.cat((pooled_output,disposition),1)
    certainty = self.op_certainty(features)
    temporality = self.op_temporality(features)
    action= self.op_action(features)
    actor= self.op_actor(features)
        
    outputs = (disposition,certainty,temporality,action,actor)
    
    return outputs
# input_path = "C:\\Users\\girid\\Documents\\CMED21\\M2\\data\\"
# full_df= pd.read_csv(input_path+"Spacy_all_columns_TRAIN.csv")
# valid_df = pd.read_csv(input_path+"Spacy_all_columns_VALIDATION.csv")

#Full train 
input_path = "C:\\Users\\girid\\Documents\\CMED21\\M2\\data\\typed_markers\\"
full_df = pd.read_csv(input_path+"Full_Train.csv")
valid_df = pd.read_csv(input_path+"Spacy_TEST_dataset.csv")


# print(full_df.head(10))
print(full_df.columns)
disposition_task_labels = list(set(list(full_df['disposition'])))
NUM_LABELS = len(disposition_task_labels)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
op_path = r"output\\spacy\\sliding_window\\test_set_results\\hierarchical_multitask\\"

def main():

  class Logger(object):
    def __init__(self):
      self.terminal = sys.stdout
      self.log = open(op_path+"logfile_base_shared_all_tasks.log","a+")

    def write(self, message):
      self.terminal.write(message)
      self.log.write(message)

    def flush(self):
      #this flush method is needed for python 3 compatibility.
      #this handles the flush command by doing nothing.
      #you might want to specify some extra behavior here.
      pass    
    
  sys.stdout = Logger()
  SEED = 41
  torch.manual_seed(SEED)
  torch.cuda.manual_seed(SEED)
  # torch.backends.cudnn.deterministic = True

  try_models = {}
  try_models['clinical_bert'] = "emilyalsentzer/Bio_ClinicalBERT"

  label= ['certainty','temporality','action','actor','disposition']
  sub_only_disp_df = full_df.copy()
  valid_only_disp = valid_df.copy()
  def ret_labels(sub_only_disp_df,l):
    le = preprocessing.LabelEncoder()
    train_labels = sub_only_disp_df[l].tolist()
    val_labels = valid_sub_only_disp_df[l].tolist()

    yt = le.fit_transform(train_labels)
    yv = le.transform(val_labels)
    task_labels = le.classes_
    NUM_LABELS = len(task_labels)
    task_det = {"train_labels":yt, "valid_labels":yv,"num_labels":NUM_LABELS,"task_labels":task_labels}
    return task_det

  for l in label:
    # print("------- ",l,"----------")
    # print(sub_only_disp_df[l].value_counts())
    
    sub_only_disp_df = sub_only_disp_df[sub_only_disp_df.groupby(l)[l].transform('count')>40]
    sub_only_disp_df.reset_index(drop= True, inplace = True)

    valid_sub_only_disp_df = valid_only_disp[valid_only_disp.groupby(l)[l].transform('count')>12]
    valid_sub_only_disp_df.reset_index(drop= True, inplace = True)
    valid_only_disp = valid_sub_only_disp_df
  
  print("Shape after filtering for all minority classes:",sub_only_disp_df.shape)
  print("Shape after filtering for all VALID minority classes:",valid_sub_only_disp_df.shape)
  
  sentence_types = ['sentence','sent_with_identifier','sent_with_tags']
  # sentence
  train_text = list(sub_only_disp_df['sentence'])
  val_text = list(valid_sub_only_disp_df['sentence'])

  # # sentence with identifiers
  # train_text = list(sub_only_disp_df['sent_with_identifier'])
  # val_text = list(valid_sub_only_disp_df['sent_with_identifier'])

  # #sentence with typed makers
  # train_text = list(sub_only_disp_df['sent_with_tags'])
  # val_text = list(valid_sub_only_disp_df['sent_with_tags'])

  disposition_task = ret_labels(sub_only_disp_df,"disposition")
  certainty_task = ret_labels(sub_only_disp_df,"certainty")
  temporality_task = ret_labels(sub_only_disp_df, "temporality")
  action_task = ret_labels(sub_only_disp_df, "action")
  actor_task = ret_labels(sub_only_disp_df,"actor")
  all_task_det = {'disposition':disposition_task,'certainty':certainty_task,"temporality":temporality_task,"action":action_task,
  "actor":actor_task}
  train_medications = list(sub_only_disp_df['medication'])
  valid_medications = list(valid_sub_only_disp_df['medication'])

  train_sent_startchar = list(sub_only_disp_df['sentence_start'])
  valid_sent_startchar = list(valid_sub_only_disp_df['sentence_start'])
  
  train_sent_endchar = list(sub_only_disp_df['sentence_end'])
  valid_sent_endchar = list(valid_sub_only_disp_df['sentence_end'])

  train_med_start = list(sub_only_disp_df['start index'])
  train_med_end = list(sub_only_disp_df['end index'])

  valid_med_start= list(valid_sub_only_disp_df['start index'])
  valid_med_end = list(valid_sub_only_disp_df['end index']) 

  
  #creating sliding window for medication context
  def padding(encoding, attention_mask,med_mask,med_len, max_length=128, pad_value=0):
      sequence_length = len(encoding)

      if sequence_length > max_length: #selecting only longer sequences
        #where is the medication
        if (1 in med_mask): #if medication exists
          med_loc = med_mask.index(1)
          if (med_loc - int(max_length/2)) <=0: # if med is closer to the beginning
            start = 0
            end = max_length
            

          elif (med_loc+int(max_length/2)) > sequence_length: # if med is closer to the end
            end = sequence_length
            start = sequence_length - max_length
          else :        # if med is neither close to end or the beginning
            start = int(med_loc - max_length/2)
            end = int(med_loc +  max_length/2 )
          
        else:
          start = 0
          end = max_length

        input_encoding = encoding[start:end]
        attn_mask = attention_mask[start:end]
        medi_mask = med_mask[start:end]

        
          
      else:
          input_encoding =  encoding + [pad_value]*(max_length - sequence_length)
          attn_mask = attention_mask + [pad_value]*(max_length - sequence_length)
          medi_mask = med_mask + [pad_value]*(max_length - sequence_length)
      
      assert len(input_encoding) == max_length
      assert len(attn_mask) == max_length
      assert len(medi_mask) == max_length
      
      return input_encoding, attn_mask, medi_mask
  
  def encode_tokens(tokens, tokenizer):
      CLS = tokenizer.convert_tokens_to_ids('[CLS]')
      SEP = tokenizer.convert_tokens_to_ids('[SEP]')
      UNK = tokenizer.convert_tokens_to_ids('[UNK]')

      encoding = []
      token_indices = []

      encoding.append(CLS)
      for i, token_phrase in enumerate(tokens):
          
          token_encoding = tokenizer.encode(token_phrase, add_special_tokens=False)

          if not token_encoding:
              token_encoding = [UNK]

          token_start = len(encoding)
          token_end = token_start + len(token_encoding)
          token_indices.append((token_start, token_end))

          encoding.extend(token_encoding)

      encoding.append(SEP)

      return (encoding, token_indices) 
  
  def find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
      if l[ind:ind+sll]==sl:
          return (ind,ind+sll-1)
  
  
  def encode_one_sentence(i,sentence,tokenizer,medication,start_end,wt):
      
      med_wps = tokenizer.convert_ids_to_tokens(tokenizer(medication,add_special_tokens=False)['input_ids'])
      med_wps_ids = tokenizer(medication,add_special_tokens=False)['input_ids']
      med_len = len(med_wps_ids)
      output = {}
      output['input_ids'] = []
      output['attention_mask'] = []
      output['med_mask'] = []
      spacy_op = wt(sentence)
      tokens = [str(x) for x in spacy_op]
      token_lens = [len(l) for l in tokens]

      encoding, token_indices = encode_tokens(tokens, tokenizer)
      wps = tokenizer.convert_ids_to_tokens(encoding)
      med_mask= [0] * len(encoding)
      attention_mask = [1]*len(encoding)

      mark= find_sub_list(med_wps_ids,encoding)
      if mark == None:
          ids_mistake.append(i)
          
      else:
          for j in range(mark[0],mark[1]+1):
              med_mask[j] = 1
          
      max_length = 128
      input_ids,attention_mask,med_mask = padding(encoding, attention_mask,med_mask,med_len, max_length=max_length, pad_value=0)

      assert len(input_ids) == len(attention_mask)
      assert len(input_ids) == len(med_mask)

      output['med_mask'] = med_mask
      output['input_ids'] = input_ids
      output['attention_mask'] = attention_mask

      return output

  
  all_labels_train = torch.tensor(list(zip(disposition_task['train_labels'],certainty_task['train_labels'],
                      temporality_task['train_labels'],action_task['train_labels'],actor_task['train_labels'])))
  all_labels_val = torch.tensor(list(zip(disposition_task['valid_labels'],certainty_task['valid_labels'],
                      temporality_task['valid_labels'],action_task['valid_labels'],actor_task['valid_labels'])))
        

  output_dims = {'disposition': disposition_task['num_labels'],'certainty':certainty_task['num_labels'],
  "temporality":temporality_task['num_labels'],
  "action":action_task['num_labels'],
  "actor":actor_task['num_labels']}

  # create torch dataset with the encodings
  class CMED_data(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
      self.encodings = encodings
      self.labels  = labels

    def __getitem__(self, idx):
      item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
      item['labels'] = self.labels[idx]
      return item
    
    def __len__(self):
      return len(self.labels)
  nlp = spacy.load(r"C:\\Users\\girid\Documents\\CMED21\\scispacy\\dist\\en_core_sci_md-0.4.0\\en_core_sci_md\\en_core_sci_md-0.4.0\\")

  for m_type in try_models:
    print("============",m_type,"====================")
    tokenizer = AutoTokenizer.from_pretrained(try_models[m_type])
    output_sent_train = {'input_ids':[],'attention_mask':[],'med_mask':[]}
    output_sent_valid = {'input_ids':[],'attention_mask':[],'med_mask':[]}

    print("encoding sentences with med_mask")
    for i,text in enumerate(train_text):
        start_end = [train_sent_startchar[i],train_sent_endchar[i],train_med_start[i],train_med_end[i]]
        encoded_sentence = encode_one_sentence(i,text,tokenizer,train_medications[i],start_end,nlp)
        for k, v in encoded_sentence.items():
            output_sent_train[k].append(v)
    
    
            
    for i,text in enumerate(val_text):
        start_end = [valid_sent_startchar[i],valid_sent_endchar[i],valid_med_start[i],valid_med_end[i]]
        encoded_sentence = encode_one_sentence(i,text,tokenizer,valid_medications[i],start_end,nlp)
        for k, v in encoded_sentence.items():
            output_sent_valid[k].append(v)
    print("encoding complete")




    train_dataset = CMED_data(output_sent_train, all_labels_train)
    val_dataset = CMED_data(output_sent_valid, all_labels_val)
    hidden_dropout_prob = 0.2

    # load default configuration
    config = AutoConfig.from_pretrained (try_models[m_type])
    # update default configuration
    config.hidden_dropout_prob = hidden_dropout_prob
    bert_model = AutoModel.from_pretrained(try_models[m_type],config= config)

    class_model = shared_learner_base(bert_model,output_dims).to(device)

    SAVE_MODEL_NAME = "shared_learner_all_tasks"+m_type
    LABELS = {'disposition':disposition_task['task_labels'],'certainty':certainty_task['task_labels'],'temporality':temporality_task['task_labels'],'action':action_task['task_labels'],'actor':actor_task['task_labels']}
    LEARNING_RATE = 5e-5
    TRAIN_BATCH_SIZE = 64
    EVAL_BATCH_SIZE = 64
    LOG_INTERVAL = 10
    WEIGHT_DECAY = 0.01
    EPOCHS = 20

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    optim = AdamW(bert_model.parameters(), lr=LEARNING_RATE)
    losses = defaultdict(list)
    epoch_train_loss = 0
    epoch_val_loss = 0
    loss_fct = nn.CrossEntropyLoss()

    all_losses_train ={}
    all_losses_valid = {}
    all_scores = {}
    labels_for_errors = {}
    w = (1,1,1,1,1)

    for epoch in tqdm(range(EPOCHS)):
      
      for batch in tqdm(train_loader):
          optim.zero_grad()
          input_ids = batch['input_ids'].to(device)
          attention_mask = batch['attention_mask'].to(device)
          medmask = batch['med_mask'].to(device)
          t_labels = batch['labels'].to(device)
          (disposition,certainty,temporality,action,actor)= class_model(input_ids,attention_mask=attention_mask, med_mask =medmask, labels=t_labels)
          
          all_losses_train['disposition'] = loss_fct(disposition,t_labels[:,0].clone())
          all_losses_train['certainty'] = loss_fct(certainty,t_labels[:,1].clone())
          all_losses_train['temporality'] = loss_fct(temporality,t_labels[:,2].clone())
          all_losses_train['action'] = loss_fct(action,t_labels[:,3].clone())
          all_losses_train['actor'] = loss_fct(actor,t_labels[:,4].clone())
          aggregate_loss = w[0]* all_losses_train['certainty']+ w[1] * all_losses_train['temporality']+ w[2]* all_losses_train['action']+ w[3]* all_losses_train['actor'] + w[4]* all_losses_train['disposition']
          all_losses_train['aggregate_loss'] = aggregate_loss
          
          epoch_train_loss += aggregate_loss
          aggregate_loss.backward()
          torch.nn.utils.clip_grad_norm_(class_model.parameters(), 1)
          
          optim.step()

      epoch_train_loss = epoch_train_loss/len(train_loader)
      losses['train_loss'].append(epoch_train_loss)
      class_model.eval()
      
      print("Running Validation...")
      all_pred_labels = defaultdict(list)
      all_true_labels = defaultdict(list)

      for batch in tqdm(val_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        medmask = batch['med_mask'].to(device)
        v_labels = batch['labels'].to(device)
        with torch.no_grad():
           (v_disposition,v_certainty,v_temporality,v_action,v_actor) = class_model(input_ids, attention_mask=attention_mask,med_mask =medmask, labels=v_labels)
          
        all_losses_valid['disposition'] = loss_fct(v_disposition,v_labels[:,0].clone().type(torch.LongTensor).to(device))
        all_losses_valid['certainty'] = loss_fct(v_certainty,v_labels[:,1].clone().type(torch.LongTensor).to(device))
        all_losses_valid['temporality'] = loss_fct(v_temporality,v_labels[:,2].clone().type(torch.LongTensor).to(device))
        all_losses_valid['action'] = loss_fct(v_action,v_labels[:,3].clone().type(torch.LongTensor).to(device))
        all_losses_valid['actor'] = loss_fct(v_actor,v_labels[:,4].clone().type(torch.LongTensor).to(device))
        aggregate_loss_valid = w[0]* all_losses_valid['certainty']+ w[1] * all_losses_valid['temporality']+ w[2]* all_losses_valid['action']+ w[3]* all_losses_valid['actor'] + w[4]* all_losses_valid['disposition']
        all_losses_valid['aggregate_loss'] = aggregate_loss_valid
        
        epoch_val_loss += aggregate_loss_valid
        
        
        softmax= torch.nn.Softmax(dim=1)
        output_probs = {}
        pred_labels = {}
        output_probs['disposition']  = softmax(v_disposition)
        output_probs['certainty']= softmax(v_certainty)
        output_probs['temporality']= softmax(v_temporality)
        output_probs['action']= softmax(v_action)
        output_probs['actor']= softmax(v_actor)

        pred_labels['disposition'] = torch.argmax(output_probs['disposition'], dim=-1).tolist()
        pred_labels['certainty'] = torch.argmax(output_probs['certainty'], dim=-1).tolist()
        pred_labels['temporality'] = torch.argmax(output_probs['temporality'], dim=-1).tolist()
        pred_labels['action'] = torch.argmax(output_probs['action'], dim=-1).tolist()
        pred_labels['actor'] = torch.argmax(output_probs['actor'], dim=-1).tolist()

        all_pred_labels['disposition'].extend(pred_labels['disposition'])
        all_true_labels['disposition'].extend(v_labels[:,0].squeeze().tolist())

        all_pred_labels['certainty'].extend(pred_labels['certainty'])
        all_true_labels['certainty'].extend(v_labels[:,1].squeeze().tolist())

        all_pred_labels['temporality'].extend(pred_labels['temporality'])
        all_true_labels['temporality'].extend(v_labels[:,2].squeeze().tolist())

        all_pred_labels['action'].extend(pred_labels['action'])
        all_true_labels['action'].extend(v_labels[:,3].squeeze().tolist())

        all_pred_labels['actor'].extend(pred_labels['actor'])
        all_true_labels['actor'].extend(v_labels[:,4].squeeze().tolist())

      scores ={}
      scores['disposition'] = [classification_report(all_true_labels['disposition'], all_pred_labels['disposition'], digits=3,  target_names=LABELS['disposition'],output_dict=True),confusion_matrix(all_true_labels['disposition'], all_pred_labels['disposition'])]
      scores['certainty'] = [classification_report(all_true_labels['certainty'], all_pred_labels['certainty'], digits=3, target_names=LABELS['certainty'],output_dict=True),confusion_matrix(all_true_labels['certainty'], all_pred_labels['certainty'])]
      scores['temporality'] = [classification_report(all_true_labels['temporality'], all_pred_labels['temporality'], digits=3,target_names=LABELS['temporality'],output_dict=True),confusion_matrix(all_true_labels['temporality'], all_pred_labels['temporality'])]
      scores['action'] = [classification_report(all_true_labels['action'], all_pred_labels['action'], digits=3,target_names=LABELS['action'],  output_dict=True),confusion_matrix(all_true_labels['action'], all_pred_labels['action'])]
      scores['actor'] = [classification_report(all_true_labels['actor'], all_pred_labels['actor'], digits=3,target_names=LABELS['actor'],output_dict=True),confusion_matrix(all_true_labels['actor'], all_pred_labels['actor'])]
      
      
      labels_for_errors[epoch] = [all_true_labels,all_pred_labels]


      all_scores[epoch] = scores
      
      # class_metrics = []
      # for cat, cat_score in scores.items():
      #     if cat in LABELS:
      #         class_metrics.append((cat, cat_score))
      # conf_matrix = confusion_matrix(all_true_labels, all_pred_labels)

      with open(op_path+"cr_"+"clinical_bert"+"_"+"disposition"+".txt","a+") as cr:
        print(scores, file = cr)
        # print(scores)
      
  with open(op_path+'all_scores_hierarch.pkl','wb') as f:
    pickle.dump(all_scores,f)
  
  with open(op_path+'all_labels_for_errors_hierarch.pkl','wb') as f:
    pickle.dump(labels_for_errors,f)
if __name__ == "__main__":
  main()












































