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
include "SymbolManager.lua"
include "utils.lua"

function create_vocab_file()
	print('creating vocab file')
	vocab_s = {}
	vocab_t = {}
	local f = torch.DiskFile(path.join(opt.data_dir,opt.train))
	f:quiet()
	f:clearError()
	local rawdata = f:readString('*l')
	while (not f:hasError()) do
		local l_list = rawdata:split('\t')
		for key,value in pairs(l_list[1]:strip():split(' ')) do
			if vocab_s[value] ~= nil then
				vocab_s[value] = vocab_s[value] +1
			else
				vocab_s[value] = 1
			end
		end
		for key,value in pairs(l_list[2]:strip():split(' ')) do
			if vocab_t[value] ~= nil then
				vocab_t[value] = vocab_t[value] +1
			else
				vocab_t[value] = 1
			end
		end
		rawdata = f:readString('*l')
	end
	 -- write into file
	 file1 = torch.DiskFile(path.join(opt.data_dir,"vocab_s.txt"), "w")

	 for key, value in pairs(vocab_s) do
	 	file1:writeString(key .. "\t" .. value .. "\n")
	 end
	 file1:close()
	 file2 = torch.DiskFile(path.join(opt.data_dir,"vocab_t.txt"), "w")
	 for key, value in pairs(vocab_t) do
	 	file2:writeString(key .. "\t" .. value .. "\n")
	 end
	 file2:close()

end

function printTable(t)
    for key,value in pairs(t) do
        print(value .. " ")
    end
    print("\n")
end

function process_train_data(opt)
	
	-- create the ocab file from the training file
	create_vocab_file(opt)
	traindata = {}
	traindata.data = {}
	traindata.label = {}

	local word_manager = SymbolManager(true)
	word_manager:init_from_file(path.join(opt.data_dir,'vocab_s.txt'))
	local form_manager = SymbolManager(true)
	form_manager:init_from_file(path.join(opt.data_dir,'vocab_t.txt'))
	print('loading text file....')
	local f = torch.DiskFile(path.join(opt.data_dir,opt.train))
	f:quiet()
	f:clearError()
	local rawdata = f:readString('*l')
	while (not f:hasError()) do
		local l_list = rawdata:strip():split('\t')
		local w_list = word_manager:get_symbol_idx_for_list(l_list[1]:strip():split(' '))
		local t_list = form_manager:get_symbol_idx_for_list(l_list[2]:strip():split(' '))
		table.insert(traindata.data,w_list)
		table.insert(traindata.label,t_list)
		rawdata = f:readString('*l')
	end
	f:close()

	-- save output preprocessed files
  	local out_mapfile = path.join(opt.data_dir, 'map.t7')
  	print('saving ' .. out_mapfile)
  	torch.save(out_mapfile, {word_manager, form_manager})
end

cmd = torch.CmdLine()
cmd:option('-data_dir','./data','Data directory')
cmd:option('-train','train','training file')

opt = cmd:parse(arg)

process_train_data(opt)

--save processed data file
local datafile = path.join(opt.data_dir,opt.train .. '.t7')
torch.save(datafile,traindata)
