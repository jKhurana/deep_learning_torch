
local SymbolManager = torch.class('SymbolManager')

function SymbolManager:__init(add_special_tag)
	self.symbol2idx = {}
	self.idx2symbol = {}
	self.vocab_size = 0
	self.add_special_tag = add_special_tag

	if add_special_tag then
		-- start symbol
		self:add_symbol('<S>')
		--end symbol
		self:add_symbol('<E>')
	end
end

function SymbolManager:add_symbol(s)
	-- body
	if self.symbol2idx[s] == nil then
		self.vocab_size = self.vocab_size+1
		self.symbol2idx[s] = self.vocab_size
		self.idx2symbol[self.vocab_size] = s
	end
end

function SymbolManager:get_symbol_idx(s)
	if self.symbol2idx[s] ==  nil then
		return 0
	end
	return self.symbol2idx[s]
end

function SymbolManager:get_idx_symbol(idx)
	-- body
	if self.idx2symbol[idx] == nil then
		return '<U>' -- return unknown id
	end
	return self.idx2symbol[idx]
end

function SymbolManager:init_from_file(file)
	local f = torch.DiskFile(file,'r',true)
	f:quiet()
	f:clearError()
	local rawdata = f:readString('*l') --  read line
	while (not f:hasError()) do
		local l_list = rawdata:strip():split('\t')
		local c = tonumber(l_list[2])
		self:add_symbol(l_list[1])
		rawdata = f:readString('*l')
	end
	f:close()
end

function SymbolManager:get_symbol_idx_for_list(l)
	local result = {}
	for i = 1, #l do
		table.insert(result,self:get_symbol_idx(l[i]))
	end
	return result
end


