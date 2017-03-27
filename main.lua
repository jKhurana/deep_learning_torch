require 'torch'
require 'os'
require 'xlua'
require 'nn'
require 'nngraph'
require 'optim'
require 'math'
require 'pl'
require('pl.stringx').import()
require 'pl.seq'
dofile "model.lua"
dofile "train.lua"
dofile "SymbolManager.lua"
dofile "function_derivative.lua"
local MinibatchLoader = require 'MiniBatchLoader'


function _main_()

	--define various options
	  local cmd = torch.CmdLine()
	  cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
	  cmd:option('-data_dir', './data', 'data path')
	  -- bookkeeping
	  cmd:option('-seed',123,'torch manual random number generator seed')
	  cmd:option('-model_dir', './model', 'output directory where model get written')
	  cmd:option('-savefile','save','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
	  cmd:option('-print_every',200,'how many steps/minibatches between printing out the loss')
	  -- model params
	  cmd:option('-rnn_size', 150, 'size of LSTM internal state')
	  cmd:option('-num_layers', 1, 'number of layers in the LSTM')
	  cmd:option('-dropout',0.5,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
	  cmd:option('-dropoutrec',0,'dropout for regularization, used after each c_i. 0 = no dropout')
	  cmd:option('-enc_seq_length',40,'number of timesteps to unroll for')
	  cmd:option('-dec_seq_length',100,'number of timesteps to unroll for')
	  cmd:option('-batch_size',20,'number of sequences to train on in parallel')
	  cmd:option('-max_epochs',160,'number of full passes through the training data')
	  -- optimization
	  cmd:option('-optimization', 'SGD', 'optimization method: 0-rmsprop 1-sgd')
	  cmd:option('-learning_rate',0.01,'learning rate')
	  cmd:option('-init_weight',0.08,'initailization weight')
	  cmd:option('-learning_rate_decay',0.98,'learning rate decay')
	  cmd:option('-learning_rate_decay_after',5,'in number of epochs, when to start decaying the learning rate')
	  cmd:option('-restart',-1,'in number of epochs, when to restart the optimization')
	  cmd:option('-decay_rate',0.95,'decay rate for rmsprop')

	  cmd:option('-grad_clip',5,'clip gradients at this value')
	  cmd:text()
	  opt = cmd:parse(arg)

  -- load meta data
	local word_manager, form_manager = table.unpack(torch.load(path.join(opt.data_dir, 'map.t7')))
	print(word_manager.vocab_size)
	print(form_manager.vocab_size)
	
  -- create a model
  	local model = create_singlelayer_encoder_decoder_arch(word_manager.vocab_size,form_manager.vocab_size,opt)
  	-- set the parameters of the model
	parameters = model.gradParameters
	gradParameters = model.gradParameters
  	--graph.dot(model.encoder.fg, 'model.encoder')
  	
  	train_loader = MinibatchLoader.create(opt, 'train')

  -- train the model
  	train(eval_training_encoder_decoder)
  	os.exit(1)
  -- perform testing 

  -- calculate the accuracy

end

-- call main funciton
_main_()