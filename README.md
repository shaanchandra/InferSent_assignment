# InferSent
Implement the InferSent model introduced here (https://arxiv.org/abs/1705.02364) by Conneau et.al. 

## Code

> **train.py**

  Accepts the following paramaters as arguments (or none):
	
		- model_name : (string) 'base / lstm / bilastm / bilstm_pool' (DEFAULT set to 'bilstm')
		- nli_path : (str) path for NLI data (raw data)
		- glove_path : (str) 'path for Glove embeddings (850B, 300D)'
		- lr : (int) 'Learning rate for training'
		- checkpoint_path : (str) 'Directory to save model during training'
		- outputmodelname : (str) 'Name of the saved model'	
		- bsize : (int) 'Batch size for training'
		- emb_dim : (int) 'Embedding size of word-vectors used'
		- lstm_dim : (int) 'Dimension of hidden unit of LSTM'
		- fc_dim : (int) 'Dimension of FC Layer (classifier)'
		- n_classes : (int) 'Number of classes being predicted for the task'

This function calls 
