
from classification_models import BBert,BiLSTM

"""Read data"""

import pandas as pd
from numba import jit, cuda
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import matplotlib.pyplot as plt

input_path = "C:\\Users\\girid\\Documents\\CMED21\\data\\"
full_df= pd.read_csv(input_path+"data_with_spans.csv")
# full_df = pd.read_csv(input_path+"traindata.csv")
label= ['disposition','action','certainty','temporality','actor']
print(full_df.columns)
print(full_df.shape)
print(type(full_df))
print(full_df['spans'].head(5))

"""Clinical BERT from Huggingface"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install git+https://github.com/huggingface/transformers
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
clinical_bert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

import torch
# If there's a GPU available...
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

x= full_df.dropna(subset=['spans'])
xz= full_df[full_df.isnull().any(axis=1)]
xz.shape
# print(len(full_df.spans.tolist()))
full_df.shape

# Tokenization
spans= full_df.spans.tolist()
input = tokenizer(spans,return_tensors="pt",padding=True,add_special_tokens = True,is_split_into_words=False)
# input = tokenizer(list(full_df['spans']),return_tensors="pt",padding=True)
#output= clinical_bert_model(**input)

len(list(input['input_ids'].numpy()))

full_df['tokenizedtext'] = list(input['input_ids'].numpy())
# print(full_df['tokenizedtext'][1])
len(full_df['tokenizedtext'][1])

xx=tokenizer.convert_ids_to_tokens(full_df['tokenizedtext'][1])
# print(xx)

from datetime import date
from datetime import datetime
import os

today = str(date.today())
folder_address = "C:\\Users\\girid\\Documents\\CMED21\\output\\BBert"
# dd/mm/YY H:M:S

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
print("date and time =", dt_string)	

try:
  os.mkdir(folder_address+dt_string)
  print("folder created")
except OSError as e:
  print("file exists")

output_path = folder_address+dt_string+"\\"
print(output_path)

# Get all of the model's parameters as a list of tuples.
# params = list(clinical_bert_model.named_parameters())
# print('The BERT model has {:} different named parameters.\n'.format(len(params)))
# print('==== Embedding Layer ====\n')
# for p in params[0:5]:
#     print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
# print('\n==== First Transformer ====\n')
# for p in params[5:21]:
#     print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
# print('\n==== Output Layer ====\n')
# for p in params[-4:]:
#     print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))




# check out the cross entropy in the other library torch.nn.crossentropy
# from torch.nn import functional as F
# criterion = F.cross_entropy
from torch.nn import CrossEntropyLoss
criterion = CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = criterion.to(device)

def fileexists(filename):
  if os.path.exists(filename):
    return 'a' # append if already exists
  else:
    return 'w' # make a new file if not

"""
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
import numpy as np

def multi_accuracy(fold,preds, y, filename = '', save = False):
    p = preds.argmax(dim=1).cpu().numpy() #Predictions               
    tr = y.float()[:, 0].data.cpu().numpy() #Truth
    
    #save output for last epoch
    if save == True:
      pfilename = output_path+filename+"_fold"+str(fold)+"_p.txt"
      predictionfile = open(pfilename, fileexists(pfilename))
      np.savetxt(predictionfile, p, fmt='%5d', delimiter=',')
      predictionfile.close()

      afilename = output_path+filename+"_fold"+str(fold)+"_a.txt"
      actualfile = open(afilename, fileexists(afilename))
      
      np.savetxt(actualfile, tr, fmt='%5d', delimiter=',')
      actualfile.close()
    
    tp_tn= np.sum(p == tr) #true positives and true negatives
    metrics = [tp_tn,p,tr]
    return metrics


from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

class CMED(Dataset):
  def __init__(self,fold_data):
      self.fold_data = fold_data
                                                   
  def __getitem__(self, index):
      data, target = self.fold_data[index]
      
      #set transformations here
      
      return data, target, index

  def __len__(self):
      return len(self.fold_data)


# ####train function returns loss and metrics every epoch ####


def train(X_train_text,filename,fold,model, train_tsdata, optimizer, criterion, batch_size):
    
    true_list=[]
    pred_list=[]
    text_and_pred = {}

    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    cmed_train_dataset = CMED(train_tsdata)
    train_loader = torch.utils.data.DataLoader(cmed_train_dataset, batch_size=batch_size, shuffle=True,num_workers =1)
    
    # for batch_idx, (data, target, idx) in enumerate(train_loader):
      
    #   print('Batch idx {}, dataset index {}'.format(batch_idx, idx))
    
    batch_indices = []
    
    for x_batch, y_batch,indices in train_loader:
      
      X = Variable(torch.LongTensor(x_batch)).to(device)
      y = Variable(torch.LongTensor(y_batch.long())).to(device)
      batch_indices.append(indices)
      # print(indices)
      # print(y)
      
      optimizer.zero_grad()

      # contains the class-wise probabilities for each batch
      predictions = model(X, X.shape[0])
      print(type(predictions))


      for i in range(predictions.shape[0]):

        real_index = indices.tolist()[i]
        text_and_pred[(i,real_index)] = (predictions[i],X_train_text.iloc[[real_index]])


      loss = criterion(predictions.squeeze(),y.squeeze())
      
      #squeeze Returns a tensor with all the dimensions of input of size 1 removed.
      metrics_batch = multi_accuracy(fold,predictions.squeeze(), y, '', save = False)#:multi_accuracy(predictions.squeeze(), y,learning_rate,batch_size)
      pred_list.append(list(metrics_batch[1]))
      true_list.append(list(metrics_batch[2]))

      pred_flat_list = [item for sublist in pred_list for item in sublist]
      true_flat_list = [item for sublist in true_list for item in sublist]
      
      epoch_loss += loss.item()
      epoch_acc += metrics_batch[0] #tp+tn
      
      loss.backward()        
      optimizer.step()
    
    final_epoch_loss= epoch_loss / len(train_loader)
    final_epoch_accuracy= epoch_acc / len(train_tsdata)
    f_1_macro= f1_score(true_flat_list, pred_flat_list, average='macro')
    f_1_micro = f1_score(true_flat_list, pred_flat_list, average='micro')
    loss_and_metrics_train = [final_epoch_loss,final_epoch_accuracy,f_1_macro,f_1_micro]
    
    batchindex_file_name= output_path+"batch_indices_details_train"+filename+str(fold)+".txt"
    batchindex_file = open(batchindex_file_name,fileexists(batchindex_file_name))
    print(batch_indices,file=batchindex_file)
    return loss_and_metrics_train

"""Computing loss for the validation set- Check"""

def evaluate(X_valid_text,fold,model, valid_tsdata, criterion, batch_size, filename, save):
    
  true_list= []
  pred_list= []
  epoch_loss = 0
  epoch_acc = 0
  text_and_pred_val ={}
  
  model.eval()
  cmed_val_dataset = CMED(valid_tsdata)
  valid_loader = torch.utils.data.DataLoader(cmed_val_dataset, batch_size=batch_size, shuffle=True)
  batch_val_indices = []
  
  with torch.no_grad():
    for x_batch, y_batch,indices in valid_loader:
      X_val = Variable(torch.LongTensor(x_batch)).to(device)
      y_val = Variable(torch.LongTensor(y_batch.long())).to(device)
      batch_val_indices.append(indices)
      print(indices)

      # contains the class-wise probabilities for each batch
      predictions_val = model(X_val, X_val.shape[0])


      for i in range(predictions_val.shape[0]):

        real_index = indices.tolist()[i]
        text_and_pred_val[(i,real_index)] = (predictions_val[i],X_valid_text.iloc[[real_index]])


      loss = criterion(predictions_val.squeeze(), y_val.squeeze())

      metrics_batch = multi_accuracy(fold,predictions_val.squeeze(), y_val, filename,save)

      pred_list.append(list(metrics_batch[1]))
      true_list.append(list(metrics_batch[2]))
      epoch_loss += loss.item()
      epoch_acc += metrics_batch[0]
        
  pred_flat_list = [item for sublist in pred_list for item in sublist]
  true_flat_list = [item for sublist in true_list for item in sublist]

  final_epoch_loss= epoch_loss / len(valid_loader)
  final_epoch_accuracy= epoch_acc / len(valid_tsdata)

  f_1_macro= f1_score(true_flat_list, pred_flat_list, average='macro')
  f_1_micro = f1_score(true_flat_list, pred_flat_list, average='micro')

  batchindex_file_name= output_path+"batch_indices_details_validation"+filename+str(fold)+".txt"
  batchindex_file = open(batchindex_file_name,fileexists(batchindex_file_name))
  print(batch_val_indices,file=batchindex_file)

  loss_and_metrics_validation = [final_epoch_loss,final_epoch_accuracy,f_1_macro,f_1_micro]
  text_and_pred_val_df = pd.DataFrame(list(text_and_pred_val.items()),columns = ['index','text']) 

  text_and_pred_val_df.to_csv(output_path + "error_file_fold_"+str(fold)+".csv")
  return loss_and_metrics_validation 

import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train_evaluate(X_train_text,X_val_text,fold_indices,fold,model,train_tsdata, val_tsdata,optimizer, criterion, batchsize,learningrate,filename, n_epochs=5):
  N_EPOCHS = n_epochs
  tloss = []
  tacc = []
  tf1micro=[]
  tf1macro =[]
  
  vloss = []
  vacc = []
  vf1micro=[]
  vf1macro =[]
  # print("fold_indices:{}",fold_indices)

  savepredictions = False
  best_valid_loss = float('inf')

  for epoch in range(N_EPOCHS):
      
      start_time = time.time()
      
      if epoch == N_EPOCHS-1:
        savepredictions = True
        
      
      train_metrics = train(X_train_text,filename,fold,model, train_tsdata, optimizer, criterion, batchsize)
      validation_metrics = evaluate(X_val_text,fold,model, val_tsdata, criterion, batchsize,filename,savepredictions)
      
      train_loss= train_metrics[0]
      train_acc= train_metrics[1]
      train_f1_macro= train_metrics[2]
      train_f1_micro= train_metrics[3]

      valid_loss= validation_metrics[0]
      valid_acc= validation_metrics[1]
      valid_f1_macro= validation_metrics[2]
      valid_f1_micro= validation_metrics[3]

      tloss.append(train_loss)
      tacc.append(train_acc)
      tf1macro.append(train_f1_macro)
      tf1micro.append(train_f1_micro)
      
      
      vloss.append(valid_loss)
      vacc.append(valid_acc)
      vf1macro.append(valid_f1_macro)
      vf1micro.append(valid_f1_micro)

          
      end_time = time.time()
          
      epoch_mins, epoch_secs = epoch_time(start_time, end_time)
          
      if valid_loss < best_valid_loss:
          best_valid_loss = valid_loss
          torch.save(model.state_dict(), output_path+filename+str(fold)+"-model.pt")
      
      epoch_file_name= output_path+"epoch_details"+filename+str(fold)+".txt"
      epoch_file = open(epoch_file_name,fileexists(epoch_file_name))
      if epoch == 0:
        print("Task,Fold,Epoch,Batchsize,learningrate,Train_Loss,Train Acc,ValLoss,ValAcc,ValF1-micro,ValF1-macro",file=epoch_file)
      if epoch <= (N_EPOCHS -1):

        print("Batch size:{} Task: {}".format(batchsize,filename))
        print(f'Epoch: {epoch+1:02} Epoch Time: {epoch_mins}m {epoch_secs}s')
        print('\tTrain Loss: {} Train Acc: {}%'.format(train_loss,train_acc))
        print(f'\t Val. Loss: {valid_loss:.3f} Val. Acc: {valid_acc*100:.3f}%')
        print(f'\t Val. F1 micro: {valid_f1_micro*100:.3f}% Val. F1 macro: {valid_f1_macro*100:.3f}%')

        
        print("{},{},{},{},{},{},{},{},{},{},{}".format(filename,fold,epoch+1,batchsize,learningrate,train_loss,train_acc,valid_loss,valid_acc,valid_f1_micro,valid_f1_macro),file=epoch_file)
        # print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s',file=epoch_file)
        # print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%',file=epoch_file)
        # print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%',file=epoch_file)
        

  return best_valid_loss,tloss,vloss,tacc,vacc


def Disposition_task():
  HIDDEN_DIM = 256
  OUTPUT_DIM = 3
  N_LAYERS = 1
  # BIDIRECTIONAL = True
  DROPOUT = 0.25
  hyperparameters = [HIDDEN_DIM,OUTPUT_DIM ,N_LAYERS,DROPOUT]
  print(hyperparameters)
  
  
  model = BBert(clinical_bert_model,HIDDEN_DIM,OUTPUT_DIM,N_LAYERS,DROPOUT)
  model = model.to(device)
  #=============================
  # enter input- Disposition
  batchsize= 128
  lr= 0.01
  n_epochs = 2
  splits =3
  #==========================
  skf = StratifiedKFold(n_splits=splits)
  le = preprocessing.LabelEncoder()

  X= full_df[['tokenizedtext']]
  X_text = full_df[['spans']]
  y= full_df[[label[0]]]
  skf.get_n_splits(X,y)

  #============================
  # batchsize= [32,64,128,256]
  # lr= [0.01,0.001,0.0001,0.00001]
  # n_epochs = [100,200,500]
  
  fold = 0
  random.seed(10)
  indices =[]
  l= 'disposition'
  for train_index, val_index in skf.split(X,y):
    fold+=1
    optimizer = optim.Adam(model.parameters(),lr= lr)
    optimizer2 = optim.ASGD(model.parameters())

    print("TRAIN:", train_index, "Valid:", val_index)
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    X_train_text, X_val_text = X_text.iloc[train_index], X_text.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    print(X_train_text.head())
    # print(X_train)
    indices.append([train_index,val_index])
    
    foldindex_file_name= output_path+"fold_indices_details_train"+"disposition"+str(fold)+".txt"
    foldindex_file = open(foldindex_file_name,fileexists(foldindex_file_name))
    
    #print indices from each fold to a txt file
    print(indices,file=foldindex_file)

    #use ravel
    yt = le.fit_transform(y_train.values)
    yv = le.transform(y_val.values[l])
    #check
    
    X_train.reset_index(drop= True, inplace = True)
    y_train.reset_index(drop= True, inplace = True)

    X_train_text.reset_index(drop= True, inplace = True)
    X_val_text.reset_index(drop = True,inplace= True)

    X_val.reset_index(drop= True, inplace = True)
    y_val.reset_index(drop= True, inplace = True)

    yt = torch.from_numpy(yt.reshape((y_train.shape[0],-1))).float()
    xt = torch.tensor(X_train['tokenizedtext'])
    train_tsdata = torch.utils.data.TensorDataset(xt, yt)

    yv = torch.from_numpy(yv.reshape((y_val.shape[0],-1))).float()
    xv = torch.tensor(X_val['tokenizedtext'])
    val_tsdata = torch.utils.data.TensorDataset(xv, yv)
    best_valid_loss,tloss,vloss,tacc,vacc = train_evaluate(X_train_text,X_val_text,indices,fold,model,train_tsdata, val_tsdata,optimizer, criterion, batchsize,lr,label[0],n_epochs)
    
    
    plt.title("Train and Validation Loss")
    plt.plot(tloss,label='train loss')
    plt.plot(vloss,label='validation loss')
    plt.legend()
    plt.savefig(output_path+"loss_plot"+"disposition"+str(fold)+".png")
    # plt.show()
    print(best_valid_loss,vacc)

"""to check 
- Try removing softmax
- BCE with logits loss
- Batch and length of words dimensions
- Is the code functioning on a diff dataset?

"""
def classificationreport_and_errors():
  try:
    preds = np.loadtxt(output_path+"disposition_p.txt").astype(int).tolist()
    truth = np.loadtxt(output_path+"disposition_a.txt").astype(int).tolist()
    analysisdf = pd.DataFrame(list(zip(truth, preds)), columns = ['truth', 'predictions'])
    len(analysisdf)
    print("truth \n", analysisdf['truth'].value_counts())
    print("predictions \n",analysisdf['predictions'].value_counts())
  except:
    print("prediction files not found")


  y_true= analysisdf['truth']
  y_pred= analysisdf['predictions']


  f1_disp_micro= f1_score(y_true, y_pred, average='micro')
  f1_disp_macro= f1_score(y_true, y_pred, average='macro')

  target_names = list(le.classes_)
  result_report = open(output_path+"classification_report_disp.txt",'a')
  print(metrics.classification_report(y_true, y_pred, target_names=target_names),file=result_report)


  print("f-1 score micro is:{} \n f-1 score macro is:{}".format(f1_disp_micro,f1_disp_macro))

  # disp = analysisdf[analysisdf['truth'] == 0]
  # print(disp.shape)
  # print("accuracy of disposition {}". format(len(nodisp[nodisp['predictions'] == 0])/len(nodisp)))
  # nodisp = analysisdf[analysisdf['truth'] == 1]
  # print(nodisp.shape)
  # print("accuracy of disposition {}". format(len(disp[disp['predictions'] == 1])/len(disp)))
  # undet = analysisdf[analysisdf['truth'] == 2]
  # print(undet.shape)
  # print("accuracy of undetermined {}". format(len(undet[undet['predictions'] == 2])/len(undet)))

"""Other 4 tasks"""
def sub_tasks():

  HIDDEN_DIM = 256
  N_LAYERS = 2
  BIDIRECTIONAL = True
  DROPOUT = 0.25
  only_disp_fulldf= full_df.loc[full_df['disposition'] == 'Disposition']

  print("{} \n Shape of only disp {}".format(only_disp_fulldf['certainty'].value_counts(),only_disp_fulldf.shape))
  only_disp_fulldf.columns




  label= ['certainty','temporality','action','actor']
  # label= ['action']

  for l in label:
    #Filter out records for which the class count is less than < 40
    sub_only_disp_df = only_disp_fulldf.copy()
    sub_only_disp_df = sub_only_disp_df[sub_only_disp_df.groupby(l)[l].transform('count')>40]
    sub_only_disp_df.reset_index(drop= True, inplace = True)
    print(sub_only_disp_df[l].value_counts())
    print(sub_only_disp_df.columns)
    
    #k fold cross validation(k=3)
    # enter input- subtasks
    n_splits=3
    n_epochs_new = 10
    learning_rate = 0.001
    batchsize=5

    skf = StratifiedKFold(n_splits)
    le = preprocessing.LabelEncoder()

    X= sub_only_disp_df[['tokenizedtext']]
    X_text = sub_only_disp_df[['spans']]
    y= sub_only_disp_df[[l]]
    skf.get_n_splits(X,y)
    
    
    fold = 0
    indices =[]
    for train_index, val_index in skf.split(X,y):
      #=====================================
      
      # SET BATCHSIZE AND N_EPOCHS
      while (len(train_index) % batchsize ==1) or (len(val_index) % batchsize ==1):
        batchsize+=1
      
    
      #=============================================

      print("-----------------------------------", l ,"-----------------------------------")
      fold+=1
      indices.append([train_index,val_index])
      foldindex_file_name= output_path+"fold_indices_details_train"+l+str(fold)+".txt"
      foldindex_file = open(foldindex_file_name,fileexists(foldindex_file_name))
      print(indices,file=foldindex_file)

      print("cross validation {} of {}".format(fold,n_splits))
      output_dim = y[l].nunique()
      other_model = BBert(clinical_bert_model,HIDDEN_DIM,output_dim,N_LAYERS,DROPOUT)
      other_model = other_model.to(device)
      optimizer = optim.Adam(other_model.parameters(),lr= learning_rate)
      optimizer2 = optim.ASGD(other_model.parameters())
      print("output dimension ", output_dim)
      
      X_train, X_val = X.iloc[train_index], X.iloc[val_index]
      X_train_text, X_val_text = X_text.iloc[train_index], X_text.iloc[val_index]
      y_train, y_val = y.iloc[train_index], y.iloc[val_index]

      X_train.reset_index(drop= True, inplace = True)
      # X_train_text.reset_index(drop= True, inplace = True)
      y_train.reset_index(drop= True, inplace = True)

      
      X_val.reset_index(drop= True, inplace = True)
      # X_val_text.reset_index(drop= True, inplace = True)
      y_val.reset_index(drop= True, inplace = True)

      print("xtrain =", X_train.shape, " XVal =", X_val.shape, " y_train= ", y_train.shape, " y_val=",y_val.shape)
      le = preprocessing.LabelEncoder()

      print(X_train_text.head())
      print(type(X_train_text))
      print(X_train_text.iloc[[378]])
      # use ravel
      yt = le.fit_transform(y_train[l].values)
      yv = le.transform(y_val[l].values)
      # xt = torch.from_numpy(traindata['tokenizedtext'].values).float()
      yt = torch.from_numpy(yt.reshape((y_train[l].shape[0],-1))).float()
      xt = torch.tensor(X_train['tokenizedtext'])
      train_tsdata = torch.utils.data.TensorDataset(xt, yt)
   
      yv = torch.from_numpy(yv.reshape((y_val[l].shape[0],-1))).float()
      xv = torch.tensor(X_val['tokenizedtext'])
      val_tsdata = torch.utils.data.TensorDataset(xv, yv)

      
      # def train_evaluate(model,train_tsdata, val_tsdata,optimizer, criterion, batchsize,learningrate,filename, n_epochs=5):

      best_valid_loss,tloss,vloss,tacc,vacc =  train_evaluate(X_train_text,X_val_text,indices,fold,other_model,train_tsdata, val_tsdata,optimizer, criterion, batchsize,learning_rate,l,n_epochs_new)
      plt.title("Train and Validation Loss")
      plt.plot(tloss,label='train loss')
      plt.plot(vloss,label='validation loss')
      plt.legend()
      plt.savefig(output_path+"loss_plot"+str(l)+str(fold)+".png")
      #plt.show()
      print("best validation loss is ", best_valid_loss)

"""Error Analysis"""

def read_predictions(taskname):
  try:
    preds = np.loadtxt(output_path+ taskname +"_p.txt").astype(int).tolist()
    truth = np.loadtxt(output_path+ taskname +"_a.txt").astype(int).tolist()
    analysisdf = pd.DataFrame(list(zip(truth, preds)), columns = ['truth', 'predictions'])
    print(analysisdf.shape)
    # print(analysisdf.head(5))
  except:
    print("no files exist.")
    analysisdf= pd.DataFrame(data=None)
  return analysisdf

  for l in label:
    print("-----------"+ l +"---------------")
    output_df = read_predictions(l)
    true= output_df['truth']
    pred= output_df['predictions']

    sub_only_disp_df = only_disp_fulldf.copy()
    sub_only_disp_df = sub_only_disp_df[sub_only_disp_df.groupby(l)[l].transform('count')>40]
    sub_only_disp_df.reset_index(drop= True, inplace = True)
    
    print(sub_only_disp_df.shape)
    print(sub_only_disp_df[l].value_counts())
    
    sub_only_disp_df['pred']= pred
    sub_only_disp_df['truth'] = true
    print("Truth distribution:{}".format(sub_only_disp_df['truth'].value_counts()))
    print("-----------------------")
    print("Prediction distribution:{}".format(sub_only_disp_df['pred'].value_counts()))
    print("-----------------------")
    
    #create csv file
    sub_only_disp_df.to_csv(output_path+l+"_errors.csv")
    
    
    print(accuracy_score(true, pred,normalize=True, sample_weight=None))

    # computing class level accuracies
    analysisdf = sub_only_disp_df.copy()
    sub_labels= list(sub_only_disp_df[l].unique())
    print(sub_labels)

    for i,s in enumerate(sub_labels):
      sub_class = analysisdf[analysisdf['truth'] == i]
      # print(sub_class.shape)
      print(" {} Predicted correctly:{} out of {}".format(s,len(sub_class[sub_class['pred'] == i]),len(sub_class)))
      print("accuracy of {} {}". format(s,len(sub_class[sub_class['pred'] == i])/len(sub_class)))
      i+=1
  print("-----------------------")

def main():
  print("---------Disposition--------")
  # Disposition_task()
  sub_tasks()

if __name__ == "__main__":
    main()

