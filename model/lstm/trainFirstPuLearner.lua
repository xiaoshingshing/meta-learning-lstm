local nn = require 'nn'
local t = require 'torch'
local autograd = require 'autograd'
local DirectNode = require 'autograd.runtime.direct.DirectNode'
local util = require 'util.util'
require 'optim'

function addWeight(inputs, targets, weights, k, useCUDA)
--   print(targets)
--   print(weights)
--   print(k)
--   print(useCUDA)
   local l = inputs:size(1)
   local classNum = weights:size(2)
   local labeled = k*(classNum-1)
   local unlabeled = l-labeled
   local data
   local target
   local weight
   if useCUDA then
      data = torch.CudaTensor(labeled+unlabeled*classNum, inputs:size(2), inputs:size(3), inputs:size(4))
      target = torch.CudaTensor(labeled+unlabeled*classNum)
      weight = torch.CudaTensor(labeled+unlabeled*classNum)
   else
      data = torch.DoubleTensor(labeled+unlabeled*classNum, inputs:size(2), inputs:size(3), inputs:size(4))
      target = torch.DoubleTensor(labeled+unlabeled*classNum)
      weight = torch.DoubleTensor(labeled+unlabeled*classNum)
   end
--   print(data:size())
   local index = 1
   for i=1,l do
      if targets[i]==classNum then
         for j=1, classNum do
            data[index] = inputs[i]
            target[index] = j
            weight[index] = weights[i][j]
            index = index + 1
         end
      else
         data[index] = inputs[i]
         target[index] = targets[i]
         weight[index] = 1
         index = index + 1
      end
   end

   local shuf={}
   for i=1,index-1 do
      shuf[i]={data=data[i],target=target[i],weight=weight[i]}
   end
   _ = require 'moses'
   shuf = _.shuffle(shuf)

   for i=1,index-1 do
      data[i]=shuf[i].data
      target[i]=shuf[i].target
      weight[i]=shuf[i].weight
   end

   return data,target,weight
end

return function(opt, inputs, targets, c, learningRate, momentum, totalEpoch)
--   local c1 = collectgarbage("count")
--   print("start of first pu train:",c1)
   local learner = {}
   local model = require(opt.learner)({
      nClasses=opt.nClasses.train,
      classify=true,
      useCUDA=opt.useCUDA,
      nIn=opt.nIn,
      nDepth=opt.nDepth,
      BN_momentum=opt.BN_momentum
   }) 
   local nonpu_input=torch.CudaTensor(opt.nClasses.train*opt.nTrainShot, inputs:size(2), inputs:size(3), inputs:size(4))
   local nonpu_target=torch.CudaTensor(opt.nClasses.train*opt.nTrainShot)
   local index=1
   for i = 1, targets:size(1) do
      if(targets[i] ~= opt.nClasses.train+1) then
         nonpu_input[index]=inputs[i]
	 nonpu_target[index]=targets[i]
	 index = index + 1
      end
   end
--   print(targets)
--   print(nonpu_target)

   local params, gradParams = model.net:getParameters()
  
   local config = {
      learningRate = learningRate,
      momentum = momentum
   }
 
   for epoch = 1, totalEpoch*2 do
      function feval(params)
         gradParams:zero()

         local outputs = model.net:forward(nonpu_input)
         local loss = model.criterion:forward(outputs, nonpu_target)
         local dloss_doutputs = model.criterion:backward(outputs, nonpu_target)
         model.net:backward(nonpu_input, dloss_doutputs)

         return loss, gradParams
      end
      optim.sgd(feval, params, config)
   end

   model.net.modules[#model.net.modules] = nn.Linear(800, opt.nClasses.train+1)
   model.net = util.localize(model.net,opt)
--   for epoch = 1, totalEpoch do
--      print(epoch)
--      function feval(params)
--         gradParams:zero()

--         local outputs = model.net:forward(inputs)
--         local loss = model.criterion:forward(outputs, targets)
--         local dloss_doutputs = model.criterion:backward(outputs, targets)
--         model.net:backward(inputs, dloss_doutputs)

--         return loss, gradParams
--      end
--      optim.sgd(feval, params, config)
--   end
   params, gradParams = model.net:parameters()
   local learningRates = torch.Tensor(#params):fill(learningRate)
   learningRates[#model.net.modules]=learningRate * 0.1
   local optimState = {}
   for i = 1, #params do
      table.insert(optimState, {
      learningRate = learningRates[i],
     momentum = momentum,
      })
   end
   for epoch = 1,totalEpoch do
      model.net:zeroGradParameters()
      local outputs = model.net:forward(inputs)
      local loss = model.criterion:forward(outputs,targets)
      local dloss_doutputs = model.criterion:backward(outputs,targets)
      model.net:backward(inputs, dloss_doutputs)
      for i = 1, #params do
	 local feval = function(x)
            return loss, gradParams[i]
         end
         optim.sgd(feval, params[i], optimState[i])
      end   
   end

   local outputs = model.net:forward(inputs)
   local outputSize = outputs:size(1)
   local classNum = c:size(1)+1
   local weights
   
   local sm=nn.SoftMax()
   outputs = outputs:type('torch.DoubleTensor')
   outputs = sm:forward(outputs)
   
   if opt.useCUDA then
      outputs = outputs:type('torch.CudaTensor')
      weights = torch.CudaTensor(outputSize, opt.nClasses.train+1)
   else
      weights = torch.DoubleTensor(outputSize, opt.nClasses.train+1)
   end

   for i=1, outputSize do
      local sum = 0
      for j=1, classNum-1 do
         weights[i][j]=(1-c[j])/c[j]*outputs[i][j]/outputs[i][classNum]
         sum = sum + weights[i][j]
      end
      weights[i][classNum] = 1- sum
   end

   data, target, weight = addWeight(inputs, targets, weights, opt.nTrainShot, opt.useCUDA)
   model=nil
   outputs=nil
--   local c2=collectgarbage("count")
--   print("end of first pu train:",c2)
--   collectgarbage("collect")
--   local c3=collectgarbage("count")
--   print("after collect:",c3)
   return data, target, weight
end

--return trainFirstPuLearner
