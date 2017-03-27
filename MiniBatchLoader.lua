local MinibatchLoader = {}
MinibatchLoader.__index = MinibatchLoader

function MinibatchLoader.create(opt, name)
	local  trainingData = {}
  	setmetatable(trainingData, MinibatchLoader)

  	-- load the training data
	trainingData = torch.load(path.join(opt.data_dir,'train.t7'))

	-- prepare data as per requirement


end

function MinibatchLoader:sample()
  local p = math.random(#trainingData.data)
  return trainingData.data[p], trainingData.label[p]
end

function MinibatchLoader:getDataSize()
	return #trainingData.data
end


return MinibatchLoader