--The Driver Program.
require 'nn'
require 'gnuplot'
require 'optim'
csv2tensor = require 'csv2tensor'

--print(arg[1])
assert(#arg == 4, '\n\nNeed four command line arguments. \nUsage: th run.lua <heuristic> <dataset> <activation> <p>\n\n')
require(arg[1]) --Use the appropriate heuristic
require 'dataset-mnist'

function train(opt, ds, model, criterion, tds)
	local train_losses = {}
	local test_losses = {}
	local train_error = {}
	local test_error = {}
	for i = 1, opt.epochs do
		print('-----EPOCH ', i, '-----')
		--Shuffle Data
		local shuffled_indices =  torch.randperm(opt.train_size, 'torch.LongTensor')
		ds.data = ds.data:index(1, shuffled_indices):squeeze()
		ds.labels = ds.labels:index(1, shuffled_indices):squeeze()
		local cur = 1
		local j = 0
		while cur < opt.train_size do
			local cur_ds = {data = ds.data[{{cur, math.min(cur+opt.batch_size, opt.train_size)},{}}], 
			      labels = ds.labels[{{cur, math.min(cur+opt.batch_size, opt.train_size)}}]}
			cur = cur + opt.batch_size
			j = j + 1
			------------------------------------------------------------------------
			--get parameters
			local p,g = model:parameters()
			local params, grads  = nn.Module.flatten(p), nn.Module.flatten(g)
			-- returns loss, grad
			local feval = function(x)
				if x ~= params then
					params:copy(x)
				end
				grads:zero()
				-- forward
				local outputs = model:forward(cur_ds.data)
				local loss = criterion:forward(outputs, cur_ds.labels)
				if j % opt.print_every == 0 then
					l, e = test(model, criterion, cur_ds)
					train_error[#train_error + 1] = e
					print(string.format("iteration %4d, training error = %1.6f", j, e))
				end
				-- backward
				local dloss_doutput = criterion:backward(outputs, cur_ds.labels)
				model:backward(cur_ds.data, dloss_doutput)
				return loss, grads
			end
			------------------------------------------------------------------------
			-- optimization loop
			--
			local _, loss = optim.adagrad(feval, params, opt)
			train_losses[#train_losses + 1] = loss[1] -- append the new loss
			if j % opt.print_every == 0 then
				l,e = test(model, criterion, tds) 
				test_losses[#test_losses+1] = l
				test_error[#test_error + 1] = e
				print(string.format("                training loss = %6.6f", loss[1]))
				print(string.format("                test error = %1.6f", e))
				print(string.format("                test loss = %6.6f\n", l))
			end	
		end	
		Rethink(model) --EPOCHAL RETHINK
	end
	return model, train_losses, train_error, test_losses, test_error
end

function test(model, criterion, ds)
	--Test the model.
	for i=1,#model.modules,2 do
		model.modules[i]:testing()
	end
	local outputs = model:forward(ds.data)
	local classProbabilities = torch.exp(outputs)
	local _, classPredictions = torch.max(classProbabilities, 2)
	local loss = criterion:forward(outputs, ds.labels)
	local err =  torch.ne(classPredictions:byte(), ds.labels:byte()):sum()/(#classPredictions)[1]
	for i=1,#model.modules,2 do
		model.modules[i]:training()
	end
	return loss, err
end

function Rethink(model)
	--All layers do a rethink
	for i=1,#model.modules,2 do
		model.modules[i]:Rethink()
	end
end

function plot(train_params, test_params, fname, xlabel, ylabel, title)
	--Plotting helper function
	gnuplot.pngfigure(fname)
	if train_params and test_params then
		gnuplot.plot(train_params, test_params)
	elseif train_params then
		gnuplot.plot(train_params)
	else
		gnuplot.plot(test_params)
	end
	gnuplot.xlabel(xlabel)
	gnuplot.ylabel(ylabel)
	gnuplot.title(title)
	gnuplot.plotflush()
end

local function load_dataset(file_name)
	local x,_ = csv2tensor.load(file_name)
	x[{{}, {1}}]:add(1)
	--print(x)
	local data = {}
	data.data = torch.Tensor(x[{{}, {2,3}}])
	data.labels = torch.Tensor(x[{{}, {1}}]):byte():view(x:nElement()/3)
	print(data.data:size())
	print(data)
	return data
end

-- load dataset using dataset-mnist.lua into tensors (first dim of data/labels ranges over data)
local function load_mnist_dataset(train_or_test, count)
    -- load
    local data
    if train_or_test == 'train' then
        data = mnist.loadTrainSet(count, {32, 32})
    else
        data = mnist.loadTestSet(count, {32, 32})
    end

    -- vectorize each 2D data point into 1D
    data.data = data.data:reshape(data.data:size(1), 32*32)
    data.data = data.data/255

    print('--------------------------------')
    print(' loaded dataset "' .. train_or_test .. '"')
    print('inputs', data.data:size())
    print('targets', data.labels:size())
    print('--------------------------------')

    return data
end


--Jai Sri Rama

---## ALL TUNABLE PARAMS IN ONE PLACE! ##---

assert(#arg == 4, '\n\nNeed four command line arguments. \nUsage: th run.lua <heuristic> <dataset> <activation> <p>\n\n')

local opt = {
	epochs = 5,
	batch_size = 10,
	print_every = 5,  
	train_size = 1000,
	test_size = 100,
	learningRate = 1e-1,
	activation = arg[3] or 'ReLU'
}

local opsize = {
	clusterincluster = 2,
	corners = 4,
	crescentfullmoon = 2,
	halfkernel = 2,
	outlier = 4,
	twospirals = 2,
	mnist=10
}

--The architecture.
mlp = nn.Sequential();  -- make a multi-layer perceptron

if arg[2] == 'mnist' then
	inputs = 32*32
else
	inputs = 2
end

outputs = opsize[arg[2]]; HUs = 10; p = tonumber(arg[4]) or 1; -- parameters

mlp:add(nn.Proj(inputs, HUs, p/2))
if opt.activation == 'ReLU' then
	mlp:add(nn.ReLU())
	mlp:add(nn.Proj(HUs, HUs, p))
	mlp:add(nn.ReLU())
elseif opt.activation == 'Tanh' then
	mlp:add(nn.Tanh())
	mlp:add(nn.Proj(HUs, HUs, p))
	mlp:add(nn.Tanh())	
elseif opt.activation == 'Sigmoid' then
	mlp:add(nn.Sigmoid())	
	mlp:add(nn.Proj(HUs, HUs, p))
	mlp:add(nn.Sigmoid())	
else
	error('Unknown activation!')
end

mlp:add(nn.Proj(HUs, outputs, p))
mlp:add(nn.SoftMax())
criterion = nn.ClassNLLCriterion()  

--GET THE DATA!
local train_data,test_data
if arg[2] == 'mnist' then
	opt.train_size = 10000 --Can tune as well.
	opt.test_size = 1000
	train_data = load_mnist_dataset('train', opt.train_size)
	test_data = load_mnist_dataset('test', opt.test_size)
else
	train_data = load_dataset('make_dataset/'..arg[2]..'.csv')
	test_data = load_dataset('make_dataset/'..arg[2]..'_test.csv')
	opt.train_size = train_data.data:size()[1]
	opt.test_size = test_data.data:size()[1]
end

print("Model:\n", mlp, "\n\nParameters:\n", opt)

-- TRAIN!
mlp, train_losses, train_error, test_losses, test_error = train(opt, train_data, mlp, criterion, test_data)
print('Trained!')
print(mlp, train_losses, train_error, test_losses, test_error)

--Make directory to log!
local dir_name = os.date('%B_')..os.date('%D'):sub(4,5)..'_'..arg[1]..'_'..arg[2]..'_'..arg[3]..'_p'..arg[4]
os.execute('mkdir '..dir_name)

--SERIALIZE!
torch.save(dir_name..'/mlp.dat', mlp)
torch.save(dir_name..'/train_losses.dat', train_losses)
torch.save(dir_name..'/train_error.dat', train_error)
torch.save(dir_name..'/test_losses.dat', test_losses)
torch.save(dir_name..'/test_error.dat', test_error)
--torch.save(dir_name..'/opt.dat', opt)

print('Saved model and losses!')

--PLOT!
tr_lplt = {'Training Loss',
	torch.range(1, #train_losses),        -- x-coordinates for data to plot, creates a tensor holding {1,2,3,...,#losses}
	torch.Tensor(train_losses),           -- y-coordinates (the training losses)
	'-'}
tr_eplt = {'Training Error',
	torch.linspace(1, (#train_error)*opt.print_every, #train_error),
	torch.Tensor(train_error),
	'-'
	}
tst_lplt = {'Test Loss',
	torch.linspace(1, (#test_losses)*opt.print_every, #test_losses),
	torch.Tensor(test_losses),
	'-'
	}
tst_eplt = {'Test error',
	torch.linspace(1, (#test_error)*opt.print_every, #test_error),
	torch.Tensor(test_error),
	'-'
	}

plot(tr_lplt, tst_lplt, dir_name..'/lplts.png', 'Iterations', 'Loss', 'Loss Plot')
plot(tr_eplt, tst_eplt, dir_name..'/eplts.png', 'Iterations', 'Error', 'Error Plot')

plot(tr_eplt, null, dir_name..'/tr_eplt.png', 'Iterations', 'Error', 'Training Error Plot')
plot(tr_lplt, null, dir_name..'/tr_lplt.png', 'Iterations', 'Loss', 'Training Loss Plot')

plot(null, tst_eplt, dir_name..'/tst_eplt.png', 'Iterations', 'Error', 'Test Error Plot')
plot(bull, tst_lplt, dir_name..'/tst_lplt.png', 'Iterations', 'Loss', 'Test Loss Plot')

print('Plots saved!')

