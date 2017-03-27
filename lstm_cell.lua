-- basic lstm cell
-- input; x, previsou_state
function lstm(x, prev_c, prev_h)
  -- Calculate all four gates in one go
  local i2h = nn.Linear(opt.rnn_size, 4 * opt.rnn_size)(x)
  local h2h = nn.Linear(opt.rnn_size, 4 * opt.rnn_size)(prev_h)
  local gates = nn.CAddTable()({i2h, h2h})
  
  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slize the n_gates dimension
  local reshaped_gates =  nn.Reshape(4, opt.rnn_size)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)
  
  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates)):annotate{name='in_gate'}
  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates)):annotate{name='in_transform'}
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates)):annotate{name='forget_gate'}
  local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates)):annotate{name='out_gate'}
  
  if opt.dropoutrec > 0 then
    in_transform = nn.Dropout(opt.dropoutrec)(in_transform)
  end

  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}):annotate{name='next_c_1'},
      nn.CMulTable()({in_gate, in_transform}):annotate{name='next_c_2'}
  })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)}):annotate{name='next_h'}

  return next_c, next_h
end

