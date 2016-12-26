--- drop.lua : Drop weights in the band  [µ-pσ, µ+pσ] during training. Halve the weights while testing.--
local Proj, parent = torch.class('nn.Proj', 'nn.Linear')

function Proj:__init(inputSize, outputSize, p)
	self.train = true
	self.p = p
	self.noiseWeight = torch.Tensor(outputSize, inputSize)
	self.noiseBias = torch.Tensor(outputSize)
	parent.__init(self, inputSize, outputSize)
end

function Proj:testing()
	self.train = false
	self.weight = 0.5*self.weight
end

function Proj:training()
	self.weight = 2*self.weight
	self.train = true
end

function Proj:reset(stdv)  --Initialization of parameters.  
	if stdv then
		stdv = stdv * math.sqrt(3)
	else
		stdv = 1./math.sqrt(self.weight:size(2))
	end
	if nn.oldSeed then
		for i=1,self.weight:size(1) do
			self.weight:select(1, i):apply(function()
				return torch.uniform(-stdv, stdv)
			end)
			self.bias[i] = torch.uniform(-stdv, stdv)
		end
	else
		self.weight:uniform(-stdv, stdv)
		self.bias:uniform(-stdv, stdv)
	end
	self.noiseWeight:fill(1)
	self.noiseBias:fill(1)
	return self
end

function Proj:Rethink()
	self.noiseWeight:copy(torch.lt((self.weight-torch.mean(self.weight)):abs(), self.p*torch.std(self.weight)))
	--self.noiseBias:copy(torch.lt((self.bias-torch.mean(self.bias)):abs(), self.p*torch.std(self.bias)))	
	self.noiseWeight:cmul(self.weight)
	--self.noiseBias:cmul(self.bias)
	--print('Rethink!')
end

function Proj:updateOutput(input)
	if input:dim() == 1 then
		self.output:resize(self.bias:size(1))
	--[[	if self.train then
			self.output:copy(self.noiseBias)
			self.output:addmv(1, self.noiseWeight, input)
		else
	]]
		self.output:copy(self.bias)
		self.output:addmv(1, self.weight, input)
	--	end
	elseif input:dim() == 2 then
		local nframe = input:size(1)
		local nElement = self.output:nElement()
		self.output:resize(nframe, self.bias:size(1))
		if self.output:nElement() ~= nElement then
			self.output:zero()
		end
		self.addBuffer = self.addBuffer or input.new()
		if self.addBuffer:nElement() ~= nframe then
			self.addBuffer:resize(nframe):fill(1)
		end
		--[[ if self.train then
			self.output:addmm(0, self.output, 1, input, self.noiseWeight:t())
			self.output:addr(1, self.addBuffer, self.noiseBias)
		else ]]
		self.output:addmm(0, self.output, 1, input, self.weight:t())
		self.output:addr(1, self.addBuffer, self.bias)
		--end
	else
		error('input must be vector or self.weight')
end
	return self.output
end

function Proj:updateGradInput(input, gradOutput)
	if self.gradInput then
		local nElement = self.gradInput:nElement()
		self.gradInput:resizeAs(input)
		if self.gradInput:nElement() ~= nElement then
			self.gradInput:zero()
		end
		if input:dim() == 1 then
			if self.train then
				self.gradInput:addmv(0, 1, self.noiseWeight:t(), gradOutput)
			else
				self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
			end
		elseif input:dim() == 2 then
			if self.train then
				self.gradInput:addmm(0, 1, gradOutput, self.noiseWeight)
			else
				self.gradInput:addmm(0, 1, gradOutput, self.weight)
			end
		end
		return self.gradInput
	end
end
