require 'torch'
require 'nn'
require 'nngraph'
include 'lstm_cell.lua'
include 'utils.lua'


-- This file consists of model 
-- Design your model using nngraph module and store in model parameters

-- single layer encoder with additional look up table for word embedding
function create_singlelayer_lookup_encoder(w_size,opt)
	-- input nngraph nodes
	x = nn.Identity()()
	prev_s = nn.Identity()()

	-- make lookup table
	local x_in = nn.LookupTable(w_size,opt.rnn_size)(x):annotate{name="enc_lookup"}
	local next_s = {}
	local splitted = {prev_s:split(2)}
	local next_c,next_h = lstm(x_in,splitted[1],splitted[2])
	table.insert(next_s,next_c)
	table.insert(next_s,next_h)

	local m = nn.gModule({x, prev_s}, {nn.Identity()(next_s)})
	return m
end

-- multilayer encoder
function create_multilayer_lookup_encoder(w_size,opt)
	-- input
	x = nn.Identity()()
	prev_s = nn.Identity()()

	-- make look up table and store it in the lowest index
	i = {[0] = nn.LookupTable(w_size,opt.rnn_size)(x):annotate{name="enc_lookup"}}
	local next_s = {}
	local splitted = {prev_s:split(2 * opt.num_layers)}

	for layer_id=1, opt.num_layers do
		local prec_c = splitted[2 * layer_id-1]
		local prev_h = splitted[2 * layer_id]
		local x_in = i[layer_id-1]
		local next_c,next_h = lstm(x_in,prec_c,prev_h)
		table.insert(next_s,next_c)
		table.insert(next_s,next_h)
		i[layer_id] = next_h
	end
	local m = nn.gModule({x,prev_s},{nn.Identity()(next_s)})

	return m

end

-- tree encoder


-- single layer decoder 
function create_singlelayer_decoder(w_size,opt)
	--input
	x = nn.Identity()()
	prev_s = nn.Identity()()

	local next_s = {}
	local splitted = {prev_s:split(2)}
	local next_c,next_h = lstm(x,splitted[1],splitted[2])
	table.insert(next_s,next_c)
	table.insert(next_s,next_h)

	-- add ouptut layer
	local h2y = nn.Linear(opt.rnn_size, w_size)(next_h)
	local pred = nn.LogSoftMax()(h2y)
	local m = nn.gModule({x,prev_s},{pred,nn.Identity()(next_s)})
	return m
end

-- multilayer decoder

--tree decoder



-- This mehthod create the encoder decoder architecture 
function create_singlelayer_encoder_decoder_arch(wsize_s,wsize_t,opt)
	model = {}
	-- add encoder and decoder
	model.encoder = create_singlelayer_lookup_encoder(wsize_s,opt)
	model.decoder = create_singlelayer_decoder(wsize_t,opt)

	-- define model criterion
	model.criterion = nn.ClassNLLCriterion()

	-- define the encoder and decoder state
	model.enc_s={}
	model.dec_s={}

	table.insert(model.enc_s,torch.Tensor(opt.rnn_size))
	table.insert(model.enc_s,torch.Tensor(opt.rnn_size))

	-- define the ouput
	model.prediction = {}

	-- collect the parameters and initialize them
	model.parameters,model.gradParameters = combine_all_parameters(model.encoder,model.decoder)
	model.parameters:uniform(-opt.init_weight,opt.init_weight)

	-- set model to training mode(for module that differ in training and testing, like Dropout)
	model.encoder:training()
	model.decoder:training()


	return model

end

--tree decoder



