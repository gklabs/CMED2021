import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss

criterion = CrossEntropyLoss()


class BBert(nn.Module):
    def __init__(self,bert,hidden_dim,output_dim,n_layers,dropout):
        super().__init__() #call super class constructor for nn.module
        print("================================================================") 
        
        self.bert = bert
        self.hidden_size = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        
        #go into BERT's embedding func
        self.embedding_dim = bert.config.to_dict()['hidden_size']
        
        # self.bidirectional = bidirectional
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers = n_layers, bidirectional=bidirectional)
        # self.rnn = nn.GRU(embedding_dim,hidden_dim,num_layers = n_layers,bidirectional = bidirectional,batch_first = True,
                          # dropout = 0 if n_layers < 2 else dropout)
        
        self.relu = nn.ReLU()
        # self.fc1 = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 256)
        
        self.fc1= nn.Linear(self.embedding_dim,self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_dim)
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(dropout)

    
    # def init_hidden(self, biflag, batch_size):
    #     return(Variable(torch.zeros(biflag, batch_size, self.hidden_size)).to(device),
    #                     Variable(torch.zeros(biflag, batch_size, self.hidden_size)).to(device))
       
    def forward(self, text, batch_size):
        with torch.no_grad():
            embedded = self.bert(text) #calling bio-bert 
        
        
        embedding_recast= embedded[0][:,0,:]#.numpy()
        # embedded = embedded.permute(1, 0, 2)      # change order of output from embedding layer  
        
        # print("embedding_recast:",embedding_recast.shape)
        layer1= self.fc1(embedding_recast)
        # print("layer1shape:",layer1.shape)


        layer1_activation= self.relu(layer1)
        drop = self.dropout(layer1_activation)
        
        layer2= self.relu(self.fc2(drop))
        # print("layer2shape:",layer2.shape)
        final_output = layer2
        # final_output = self.softmax(layer2)
        
        
        
        # print("last layer:",final_output.shape)
        
        # print("layer1shape:",layer1.shape)
        # print("layer2shape:",layer2.shape)
        # print("last layer:",final_output.shape)
        
        # x = x.view(x.size(0), -1) 
        # if(self.bidirectional == True):
        #     self.hidden = self.init_hidden(2 * self.n_layers, batch_size)
        # else:
        #     self.hidden = self.init_hidden(1 * self.n_layers, batch_size)       

        # output, (final_hidden_state, final_cell_state) = self.lstm(embedded, self.hidden)

        # #hidden = [n layers * n directions, batch size, emb dim]

        # relu = self.relu(final_hidden_state[-1])      
        # dense1 = self.fc1(relu)
        # drop = self.dropout(dense1)       
        # final_output = self.softmax(self.fc2(drop))
        
        return final_output
######################## Bio Clinical BERT with LSTM layer #########################

class BiLSTM(nn.Module):
    def __init__(self,bert,hidden_dim,output_dim,n_layers,bidirectional,dropout):
        super().__init__()
        print("================================================================") 
        self.bert = bert
        self.hidden_size = hidden_dim * 2 if bidirectional else hidden_dim
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.lstm = nn.LSTM(embedding_dim, self.hidden_size, num_layers = n_layers, bidirectional=bidirectional)
        # self.rnn = nn.GRU(embedding_dim,hidden_dim,num_layers = n_layers,bidirectional = bidirectional,batch_first = True,
        #                   dropout = 0 if n_layers < 2 else dropout)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, output_dim)
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(dropout)

    
    # def init_hidden(self, biflag, batch_size):
    #     return(Variable(torch.zeros(biflag, batch_size, self.hidden_size)).to(device),
    #                     Variable(torch.zeros(biflag, batch_size, self.hidden_size)).to(device))
       
    def forward(self, text, batch_size):
        with torch.no_grad():
            embedded = self.bert(text)[0] #calling bio-bert 
        embedded = embedded.permute(1, 0, 2)      # change order of    
        if(self.bidirectional == True):
            self.hidden = self.init_hidden(2 * self.n_layers, batch_size)
        else:
            self.hidden = self.init_hidden(1 * self.n_layers, batch_size)       

        output, (final_hidden_state, final_cell_state) = self.lstm(embedded, self.hidden)
        #hidden = [n layers * n directions, batch size, emb dim]

        relu = self.relu(final_hidden_state[-1])      
        dense1 = self.fc1(relu)
        drop = self.dropout(dense1)       
        final_output = self.softmax(self.fc2(drop))
        
        return final_output
################################ Multi-task learner with Base Bio-BERT and two hidden linear layers

class MTL_BBert(nn.Module):
    def __init__(self,bert,hidden_dim,output_dim,n_layers,dropout,prev_weights):
        super().__init__() #call super class constructor for nn.module
        print("================================================================") 
        
        self.bert = bert
        self.hidden_size = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        
        #go into BERT's embedding func
        self.embedding_dim = bert.config.to_dict()['hidden_size']
        self.relu = nn.ReLU()
        
        self.fc1= nn.Linear(self.embedding_dim,self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_dim)
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(dropout)

    
    # def init_hidden(self, biflag, batch_size):
    #     return(Variable(torch.zeros(biflag, batch_size, self.hidden_size)).to(device),
    #                     Variable(torch.zeros(biflag, batch_size, self.hidden_size)).to(device))
       
    def forward(self, text, batch_size):
        with torch.no_grad():
            embedded = self.bert(text) #calling bio-bert 
        
        
        embedding_recast= embedded[0][:,0,:]#.numpy()
        # embedded = embedded.permute(1, 0, 2)      # change order of output from embedding layer  
        
        # print("embedding_recast:",embedding_recast.shape)
        layer1= self.fc1(embedding_recast)
        # print("layer1shape:",layer1.shape)


        layer1_activation= self.relu(layer1)
        drop = self.dropout(layer1_activation)
        
        layer2= self.relu(self.fc2(drop))
        # print("layer2shape:",layer2.shape)
        final_output = layer2
        # final_output = self.softmax(layer2)
        
        
        
        # print("last layer:",final_output.shape)
        
        # print("layer1shape:",layer1.shape)
        # print("layer2shape:",layer2.shape)
        # print("last layer:",final_output.shape)
        
        # x = x.view(x.size(0), -1) 
        # if(self.bidirectional == True):
        #     self.hidden = self.init_hidden(2 * self.n_layers, batch_size)
        # else:
        #     self.hidden = self.init_hidden(1 * self.n_layers, batch_size)       

        # output, (final_hidden_state, final_cell_state) = self.lstm(embedded, self.hidden)

        # #hidden = [n layers * n directions, batch size, emb dim]

        # relu = self.relu(final_hidden_state[-1])      
        # dense1 = self.fc1(relu)
        # drop = self.dropout(dense1)       
        # final_output = self.softmax(self.fc2(drop))
        
        return final_output

############## Shared learning setup ####################

class shared_learner(nn.Module):
    def __init__(self,bert,hidden_dim,output_dim_actor,output_dim_certainty,output_dim_temporality,output_dim_action,n_layers,dropout):
        super().__init__() #call super class constructor for nn.module
        print("================================================================") 
        
        self.bert = bert
        self.hidden_size = hidden_dim
        self.n_layers = n_layers

        self.output_dim_actor = output_dim_actor
        self.output_dim_certainty = output_dim_certainty
        self.output_dim_temporality =output_dim_temporality
        self.output_dim_action = output_dim_action


        #go into BERT's embedding func
        self.embedding_dim = bert.config.to_dict()['hidden_size']
        self.relu = nn.ReLU()
        
        #Common layer drawing off from the embedding layer
        self.fc1= nn.Linear(self.embedding_dim,self.hidden_size)

        # creating separate fully connected layers for each task
        self.fc_actor = nn.Linear(self.hidden_size, self.output_dim_actor)
        self.fc_certainty = nn.Linear(self.hidden_size,self.output_dim_certainty)
        self.fc_temporality = nn.Linear(self.hidden_size,self.output_dim_temporality)
        self.fc_action = nn.Linear(self.hidden_size,output_dim_action)


        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(dropout)

    
    # def init_hidden(self, biflag, batch_size):
    #     return(Variable(torch.zeros(biflag, batch_size, self.hidden_size)).to(device),
    #                     Variable(torch.zeros(biflag, batch_size, self.hidden_size)).to(device))
       
    def forward(self, text, batch_size):
        with torch.no_grad():
            embedded = self.bert(text) #calling bio-bert 
        
        
        embedding_recast= embedded[0][:,0,:]#.numpy()
        # embedded = embedded.permute(1, 0, 2)      # change order of output from embedding layer  
        
        # print("embedding_recast:",embedding_recast.shape)
        layer_common= self.fc1(embedding_recast)
        # print("layer1shape:",layer1.shape)

        layer1_activation= self.relu(layer_common)
        drop_common = self.dropout(layer1_activation)
        
        actor_layer_output= self.relu(self.fc_actor(drop_common))
        certainty_layer_output= self.relu(self.fc_certainty(drop_common))
        temporality_layer_output = self.relu(self.fc_temporality(drop_common))
        action_layer_output = self.relu(self.fc_action(drop_common))


        return actor_layer_output,certainty_layer_output,temporality_layer_output,action_layer_output
