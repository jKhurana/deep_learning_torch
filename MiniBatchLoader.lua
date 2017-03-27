local MinibatchLoader = {}
MinibatchLoader.__index = MinibatchLoader

function MinibatchLoader.create(opt, name)
	local  self = {}
  	setmetatable(self, MinibatchLoader)

  	-- load the training data
	self.trainingData = torch.load(path.join(opt.data_dir,'train.t7'))

	-- prepare data as per requirement

	return self

end

function MinibatchLoader:sample()
  local p = math.random(#(self.trainingData.data))
  return self.trainingData.data[p], self.trainingData.label[p]
end

function MinibatchLoader:getDataSize()
	return #(self.trainingData.data)
end


return MinibatchLoader