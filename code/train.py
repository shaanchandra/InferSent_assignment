from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


import os
import sys
import time
import argparse

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn

from data import get_nli, build_vocab, get_batch_from_idx
from models import Classifier, LSTM, biLSTM, LSTM_main
from dev_test_evals import model_eval


# Default params
LEARNING_RATE_DEFAULT = 0.1
BATCH_SIZE_DEFAULT = 64
EMB_DIM_DEFAULT = 300
nli_DEFAULT = './data/snli'
glove_DEFAULT = './data/glove/glove.840B.300d.txt'


OPTIMIZER_DEFAULT = 'SGD'
LSTM_DIM_DEFAULT = 2048
FC_DIM_DEFAULT = 512
N_CLASSES_DEFAULT = 3

MODEL_NAME_DEFAULT = 'bilstm'

CHECKOUT_DIR_DEFAULT = './checkout/'
DEVICE_DEFAULT = 'cuda'
device = ''
dtype = ''



parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type = str, default = MODEL_NAME_DEFAULT,
                      help='model name: base / lstm / bilastm / bilstm_pool')
parser.add_argument('--nli_path', type = str, default = nli_DEFAULT,
                      help='path for NLI data (raw data)')
parser.add_argument('--glove_path', type = str, default = glove_DEFAULT,
                      help='path for Glove embeddings (850B, 300D)')
parser.add_argument('--lr', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate for training')
parser.add_argument('--checkpoint_path', type = str, default = CHECKOUT_DIR_DEFAULT,
                      help='Directory of check point')
parser.add_argument('--bsize', type = int, default = BATCH_SIZE_DEFAULT,
                      help='batch size for training"')
parser.add_argument('--emb_dim', type = int, default = EMB_DIM_DEFAULT,
                      help='dimen of word embeddings used"')
parser.add_argument('--lstm_dim', type = int, default = LSTM_DIM_DEFAULT,
                      help='dimen of hidden unit of LSTM"')
parser.add_argument('--fc_dim', type = int, default = FC_DIM_DEFAULT,
                      help='dimen of FC layer"')
parser.add_argument('--n_classes', type = int, default = N_CLASSES_DEFAULT,
                      help='number of classes"')
parser.add_argument("--outputmodelname", type=str, default= 'model.pickle', 
                   help = 'saved model name')
FLAGS, unparsed = parser.parse_known_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def main():
    global dtype
    
    dtype = torch.FloatTensor
    
    #Print Flags
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


main() 

nli_path = nli_DEFAULT
glove_path = glove_DEFAULT




train, dev, test = get_nli(nli_path)
vocab, embeddings = build_vocab(train['s1']+train['s2']+test['s1']+test['s2']+dev['s1']+dev['s2'], glove_path)



config = {
'n_words'        :  len(embeddings)          ,
'emb_dim'       :  FLAGS.emb_dim   ,
'lstm_dim'       :  FLAGS.lstm_dim   ,
'dpout'          :  FLAGS.dpout    ,
'fc_dim'         :  FLAGS.fc_dim         ,
'b_size'          :  FLAGS.bsize     ,
'n_classes'      :  FLAGS.n_classes      ,
'model_name'     :  FLAGS.model_name   ,
'n_classes'      :  FLAGS.n_classes ,

}

#append every sentence with <s> in the start and </s> in the end. Also, ignore the words in sentences for which no embedding
print("\nAppending every sentence with <s> in the start and </s> in the end.")
for sent_type in ['s1', 's2']:
    for dset in ['train', 'dev', 'test']:
        eval(dset)[sent_type] = np.array([['<s>'] +
            [word for word in sent.split() if word in embeddings] +
            ['</s>'] for sent in eval(dset)[sent_type]])
print("Example sentence after processing: \n", train['s1'][0])


if FLAGS.model_name == 'base':
    classif = Classifier(config).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classif.parameters(), lr = FLAGS.lr, weight_decay = 1e-4)
elif FLAGS.model_name == 'lstm':
    
    model = LSTM_main(config).to(device)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    
#     mod_params = list(model.parameters()) + list(classif.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr = FLAGS.lr, weight_decay = 0)
    print("\n------Training uni-LSTM network---------")
    print(model)

elif FLAGS.model_name =='bilstm':
    
        
    model = LSTM_main(config).to(device)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr = FLAGS.lr, weight_decay = 0)
    print("\n------Training bi-LSTM network---------")
    print(model)
    
    
    
def base_network(epoch):
    
    print("Device = ", device)
    
    print("\n------Training base network---------")
    

    print(classif)

    print("\n-------Permuting indexes for batching------------")
    perm_idxs = np.random.permutation(len(train['label']))
    
    s1 = train['s1'][perm_idxs]
    s2 = train['s2'][perm_idxs]
    label = train['label'][perm_idxs]
    correct = 0
    train_losses = []
    logs = []
   
    n_iter = int(np.ceil(len(train['label'])/FLAGS.bsize))

    print("\n-----------Beginning Training-------------")

    classif.train()
    for i in range(n_iter):

        start = i*FLAGS.bsize
        end = start + FLAGS.bsize
        
        #Get sentence embeddings for each sentence in the batch
        s1_batch, _ = get_batch_from_idx(s1[start:end], embeddings, config)
        u = torch.sum(s1_batch, 0).to(device)

        s2_batch, _ = get_batch_from_idx(s2[start:end], embeddings, config)
        v = torch.sum(s2_batch, 0).to(device)

        label_batch = torch.LongTensor(label[start:end]).to(device)

        # prepare features from the averaged sentence representences
        features = torch.cat((u, v, torch.abs(u- v), u*v), 1).to(device)

        # forward-pass
        out = classif(features).to(device)

        #Take the arg-max as prediction (Note: torch.max returns (max, index), we take the index as our ground truth labels are (0,1,2)
        preds = torch.max(out.data,1)[1].to(device)
        
        correct += preds.long().eq(label_batch.data.long()).cpu().sum().item()

        # loss
        loss = loss_fn(out, label_batch)
        train_losses.append(loss.item())
        
#         a = list(model.parameters())[0].clone()
        # backward
        optimizer.zero_grad()
        loss.backward()
 

        # optimizer step
        optimizer.step()
#         b = list(model.parameters())[0].clone()
        
        # Train and Dev set evaluation every 1000 batches 
        if len(train_losses) == 1000:
#             print("Params equal : ", a.data - b.data)
            logs.append('{0} ; loss {1} ; accuracy train : {2}'.format(
                            start, round(np.mean(train_losses), 4),
                            round(100.*correct/(start+FLAGS.bsize), 3)))
            print(logs[-1])
           
            train_losses = []
    
    train_acc = round(100 * correct/len(s1), 2)
    print('results : epoch {0} ; mean accuracy train : {1}'
          .format(epoch, train_acc))
    
    return train_acc
    
    
    
# For the LSTM-models    
def train_network(epoch):
    
    if FLAGS.torch_device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device =torch.device('cpu')
    
    
    
    print("\n-------Permuting indexes for batching------------")

    perm_idxs = np.random.permutation(len(train['label']))
    
    s1 = train['s1'][perm_idxs]
    s2 = train['s2'][perm_idxs]
    label = train['label'][perm_idxs]
    correct = 0
    train_losses = []
    logs = []
   
    n_iter = int(np.ceil(len(train['label'])/FLAGS.bsize))

    print("\n-----------Beginning Training-------------\n")
    
    for i in range(n_iter):
        
        start = i*FLAGS.bsize
        end = start + FLAGS.bsize

        
        s1_batch, s1_lens = get_batch_from_idx(s1[start:end], embeddings, config)
        s2_batch, s2_lens = get_batch_from_idx(s2[start:end], embeddings, config)
        
        s1_batch, s2_batch = Variable(s1_batch.to(device)), Variable(s2_batch.to(device))
        label_batch = Variable(torch.LongTensor(label[start:end])).to(device)
        

        out =  model(((s1_batch, s1_lens), (s2_batch, s2_lens))).to(device)

        preds = out.data.max(1)[1].to(device)

        correct += preds.long().eq(label_batch.data.long()).cpu().sum().item()

        # loss
        loss = loss_fn(out, label_batch)
        train_losses.append(loss.item())
        
#         a = list(model.parameters())[0].clone()
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        
 
        # optimizer step
        optimizer.step()
#         b = list(model.parameters())[0].clone()
       
        

        if len(train_losses) == 1000:
#             print("Params model equal : ", torch.equal(a.data, b.data))
            logs.append('{0} ; loss {1} ; accuracy train : {2}'.format(
                            start, round(np.mean(train_losses), 2),
                            round(100.*correct/(start+FLAGS.bsize), 2)))
            print(logs[-1])
            train_losses = []
            
    train_acc = round(100 * correct/len(s1), 2)
    print('results : epoch {0} ; mean accuracy train : {1}'
          .format(epoch, train_acc))
    
    return train_acc


    
    

epoch =1
eval_old = 0

# Termination condition: Stop training when lr goes below 1e-5
# lr reduced by a factor of 0.5 everytime the previous validation accuracy is higher than the new one

while FLAGS.lr > 1e-5:
    
    if FLAGS.model_name == 'base':
        train_acc = base_network(epoch)
    else:
        train_acc = train_network(epoch)
     
    eval_acc = model_eval(epoch, 'dev', FLAGS)
    
    if eval_old > eval_acc:
        FLAGS.lr = FLAGS.lr/5
        print("learning rate changed to : ", FLAGS.lr)
    if FLAGS.lr < 1e-5:
        test_acc = model_eval(epoch, 'test', FLAGS)
        print("\n-----Training ended------")
        print("Test accuracy final = ", test_acc)
    eval_old = eval_acc    
    epoch += 1
    
    
