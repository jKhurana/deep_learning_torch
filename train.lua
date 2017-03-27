
-- Log results to files
--trainLogger = optim.Logger(paths.concat(opt.save,'train.log'))
--testLogger = optim.Logger(paths.concat(opt.save,'test.log'))

--------------------------------------- Optimizer state on the basis of optimization algorithm------------------------------------

function setOptimizerState()
	print "==> configure optimizer"
	if opt.optimization == 'CG' then
		optimState = {
			maxIter = opt.maxIter
		}
		optimMethod = optim.cg
	elseif opt.optimization == 'LBFGS' then
		optimState = {
			learningRate = opt.learningRate,
			maxIter = opt.maxIter,
			nCorrection = 10
		}
		optimMethod = optim.lbfgs

	elseif opt.optimization == 'SGD' then
		optimState = {
			learningRate = opt.learningRate,
			weightDecay = opt.weightDecay,
			momentum = opt.momentum,
			learningRateDecay = 1e-7
		}
		optimMethod = optim.sgd
	elseif opt.optimization == 'ASGD' then
	 	optimState = {
	      eta0 = opt.learningRate,
	      t0 = trsize * opt.t0
	   }
	   optimMethod = optim.asgd
	else
		error('unknown optimization method')
	end
end
----------------------------------------------------------------------------------------------------------------

----------------------------- training funciton----------------------------------

function train(m,f)
	print '==> defining training procedure'
	--set the optimizer state
	setOptimizerState(opt)
	-- local vars
	local time = sys.clock()
	
	-- Do various passes over the data
	local iterations = opt.max_epochs * train_loader:getDataSize()
	for i=1,iterations do
		--disp progress
		--xlua.progress(t,trainingData:size())

		if optimMethod == optim.asgd then
			_,_,average = optimMethod(f,parameters,optimState)
		else
			optimMethod(f,parameters,optimState)
		end
	end

	-- time taken
   	time = sys.clock() - time
   	time = time / trainData:size()
   	print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   	--update the logger
   	--trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   	
	-- save the trained model
	torch.save(string.format('%s/model.t7', opt.model_dir), model)

end