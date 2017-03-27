-------------------------- create closure to evaluate f(X) and df/dx--------
-- x ==> parameter of the model
function eval_training_encoder_decoder(x)

	local input,target = train_loader:sample()
	--get new parameters
	if x~= parameters then
			parameters:copy(x)
	end

	--reset gradient parameters
	gradParameters:zero()
---------------------------------------working till here------------------------------------



	-- initalize the encoder state(reset it for every input as inputs are independent of each other)
	model.enc_s[1]:zero()
	model.enc_s[2]:zero()
	-- calculate the f
	local f = 0
	-- feed the data into encoder
	for i = 1, #input do
		model.enc_s  = model.encoder:forward(input[i],model.enc_s)
	end
	print(model.enc_s[2])
	os.exit(1)
	-- feed the output of encoder into decoder
	model.dec_s[1]:zero()
	model.dec_s[2]:zero()

	for i=1, target:size() do
		
	end

	local err = criterion:forward(output,target)
	f = err
	-- estimate df/dx
	local df_do = criterion:backward(output,target)
	model:backward(input,df_do)

	return f,gradParameters
end