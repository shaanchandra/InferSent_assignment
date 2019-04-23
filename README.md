# InferSent
Learning general-purpose sentence representations in the natural language inference (NLI) task.

Implement the InferSent model introduced [here](https://arxiv.org/abs/1705.02364) by Conneau et.al. 
NLI is the task of classifying entailment or contradiction relationships between premises and hypotheses, such as the following:

1. *Premise* Bob is in his room, but because of the thunder and lightning outside, he cannot sleep.
2. *Hypothesis* 1 Bob is awake.
3. *Hypothesis* 2 It is sunny outside.

While the first hypothesis follows from the premise, indicated by the alignment of `cannot sleep` and `awake`, the second hypothesis contradicts the premise, as can be seen from the alignment of `sunny` and `thunder and lightning` and recognizing their incompatibility.

## Code

``` train.py```

  Accepts the following paramaters as arguments (or none as DEFAULTs for each is set already):
	
1. ***model_name*** : (string) 'base / lstm / bilastm / bilstm_pool' (DEFAULT set to 'bilstm')
2. ***nli_path*** : (str) path for NLI data (raw data)
3. ***glove_path*** : (str) 'path for Glove embeddings (850B, 300D)'
4. ***lr*** : (int) 'Learning rate for training'
5. ***checkpoint_path*** : (str) 'Directory to save model during training'
6. ***outputmodelname*** : (str) 'Name of the saved model'	
7. ***bsize*** : (int) 'Batch size for training'
8. ***emb_dim*** : (int) 'Embedding size of word-vectors used'
9. ***lstm_dim*** : (int) 'Dimension of hidden unit of LSTM'
10. ***fc_dim*** : (int) 'Dimension of FC Layer (classifier)'
11. ***n_classes*** : (int) 'Number of classes being predicted for the task'

This is the main function where the training happens and all the other modules declared in *models.py* , *data.py* and *dev_test_evals.py* are called. For further details, check the file for comments in each line and step. NOTE: to start the training process, you just need to run this file with the above mentioned paramaters (optional).


```data.py```

This is where raw SNLI data is pre-processed and word-vecs are created using the GLOVE embeddings. It involves the following modules:

1. ***get_nli()***: This reads the NLI data and partitions it into train,dev and test sets with dictionaries for 's1','s2' and labels.
2. ***build_vocab()***: This creates a mapping (word-vecs) for each word in the SNLI vocabulary to their respective word embedding in GLOVE.
3. ***get_batch_from_idx()***: This takes batches of data, pads them to equal lengths and returns the word embeddings for word in each sentence.
	

``` models.py```

This file includes all the model classes required for training and is structured as follows:
1. ***LSTM_main()***: Based on the model being trained, initializes the right class's object in itself. It encodes the sentences pairs as ***u*** and ***v*** by the chosen encoder (LSTM/biLSTM/biLSTM(max-pool) and returns the feature for the classifier in the following form:

```concatenate(u, v, |u-v|, u*v)```

2. ***LSTM()***: Encodes the provided sentences usnig a uni-directional LSTM network and returning the final state as the sentence representation.

3. ***biLSTM()***: 
	
