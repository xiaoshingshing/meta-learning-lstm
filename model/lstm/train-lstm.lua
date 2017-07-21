local t = require 'torch'
local autograd = require 'autograd'
local nn = require 'nn'
local util = require 'util.util'
local _ = require 'moses'

require 'model.lstm.meta-learner-lstm'

return function(opt, dataset)
   -- data
   local metaTrainSet = dataset.train
   local metaValidationSet = dataset.validation
   local metaTestSet = dataset.test
 
   -- keep track of errors
   local avgs = {} 
   local trainConf = optim.ConfusionMatrix(opt.nClasses.train+1)
   local valConf = {}
   local testConf = {}
   for _,k in pairs(opt.nTestShot) do
      valConf[k] = optim.ConfusionMatrix(opt.nClasses.val)
      testConf[k] = optim.ConfusionMatrix(opt.nClasses.test)
      avgs[k] = 0 
   end 

   -- learner
   local learner = getLearner(opt)  
   print("Learner nParams: " .. learner.nParams)   

   -- meta-learner     
   local metaLearner = getMetaLearner({learnerParams=learner.params, 
      nParams=learner.nParams, debug=opt.debug, 
      homePath=opt.homePath, nHidden=opt.nHidden, BN1=opt.BN1, BN2=opt.BN2})  
   local classify = metaLearner.f 
     
   -- load params from file?
   if opt.paramsFile then
      print("loading params from: " .. opt.paramsFile)
      local loadedParams = torch.load(opt.paramsFile)
      metaLearner.params = loadedParams
   end

   -- cast params to float or cuda 
   local cast = "float"
   if opt.useCUDA then
      cast = "cuda"
   end
   metaLearner.params = autograd.util.cast(metaLearner.params, cast)
   print("Meta-learner params")
   print(metaLearner.params)        

   local nEpisode = opt.nEpisode
   local cost = 0
   local timer = torch.Timer()
   local printPer = opt.printPer 
   local evalCounter = 1
   local prevIterParams

   local lstmState = {{},{}}

   ---------------------------------------------------------------------------- 
   -- meta-training

   -- init optimizer
   local optimizer, optimState = autograd.optim[opt.optimMethod](
      metaLearner.dfWithGradNorm, tablex.deepcopy(opt), metaLearner.params) 
 
   -- episode loop 
   for d=1,nEpisode do  
      -- create training epsiode
--    print("——————————")

      local c1 = collectgarbage("count")
--      print("start of first pu train:",c1)
 
      local trainSet, testSet = metaTrainSet.createEpisode({})   
   
      -- train on meta-train 
      local trainData = trainSet:get()
    
--      local trainInput, trainTarget = util.extractK(trainData.input, 
--         trainData.target, opt.nTrainShot, opt.nClasses.train)
      local trainInput = trainData.input
      local trainTarget = trainData.target

      local testData = testSet:get()
      
      local c=torch.CudaTensor(opt.nClasses.train):fill(0.2)
--      print("——————————")

      local c1 = collectgarbage("count")
--      print("start of first pu train:",c1)

      local trainInputs, trainTargets, trainWeights = require('model.lstm.trainFirstPuLearner')(opt,trainInput,trainTarget,c,1e-3,0.9,50)
      local c2 = collectgarbage("count")
--      print("end of first pu train:",c2)
      collectgarbage("collect")
      collectgarbage("collect")
      collectgarbage("collect")
      collectgarbage("collect")
      local c3 = collectgarbage("count")
--      print("after collect",c3)

--      print("33333111111111111111111111")
--      print(trainTargets)
--      print(trainWeights)
--      print("——————————")
c1 = collectgarbage("count")
--      print("start of optimize:",c1)

      local gParams, loss, prediction = optimizer(learner, trainInputs, 
         trainTargets, trainWeights, testData.input, testData.target, 
         opt.nEpochs[opt.nTrainShot],-- opt.batchSize[opt.nTrainShot])
	 1024)
      gParams={}
	 c2=collectgarbage("count")
--      print("end of optimize:",c2)
      collectgarbage("collect")
      collectgarbage("collect")
      collectgarbage("collect")
      collectgarbage("collect")
c3 = collectgarbage("count")
--      print("after collect",c3)
      cost = cost + loss      
--     print("-------------------")
     c1=collectgarbage("count")
--     print("save start:",c1)

      for i=1,prediction:size(1) do
	 prediction[i][opt.nClasses.train+1]=-100
         trainConf:add(prediction[i], testData.target[i])   
      end
     c2=collectgarbage("count")
--     print("save end:",c2)
           collectgarbage("collect")
      collectgarbage("collect")
      collectgarbage("collect")
      collectgarbage("collect")
c3 = collectgarbage("count")
--      print("after collect",c3)
      print(d)
      print(trainConf)
--printPer=1 
      -- status check of meta-training & evaluate meta-validation
      if math.fmod(d, printPer) == 0 then
         local elapsed = timer:time().real
         print(string.format(
            "Dataset: %d, Train Loss: %.3f, LR: %.3f, Time: %.4f s", 
            d, cost/(printPer), util.getCurrentLR(optimState[1]), elapsed))
         print(trainConf) 
         trainConf:zero()

         -- meta-validation loop
	 for v=1,opt.nValidationEpisode do
	
       --  for v=1,opt.nValidationEpisode do
            local trainSet, testSet = metaValidationSet.createEpisode({})
            local trainData = trainSet:get()
            local testData = testSet:get()
            
            -- k-shot loop
          --  for _,k in pairs(opt.nTrainShot) do
	  k=opt.nTrainShot
               local trainInput = trainData.input
	       local trainTarget = trainData.target
               
	       local c = torch.CudaTensor(opt.nClasses.train):fill(0.2)
	    
	       local trainInputs, trainTargets, trainWeights = require('model.lstm.trainFirstPuLearner')(opt,trainInput,trainTarget,c,1e-3,0.9,50)

               local _, prediction = classify(metaLearner.params, learner, 
                  trainInputs, trainTargets, trainWeights, testData.input, testData.target, 
                  opt.nEpochs[k] or opt.nEpochs[opt.nTrainShot], 
                  --opt.batchSize[k] or opt.batchSize[opt.nTrainShot], true)     
                  1024,true)

               for i=1,prediction:size(1) do
		  prediction[i][opt.nClasses.train+1]=-100
                  valConf[k]:add(prediction[i], testData.target[i])  
               end

          --  end
         end   
         k=opt.nTrainShot
         -- print accuracy on meta-validation set 
       --  for _,k in pairs(opt.nTrainShot) do 
            print('Validation Accuracy (' .. opt.nValidationEpisode 
               .. ' episodes, ' .. k .. '-shot)')
            print(valConf[k]) 
            valConf[k]:zero()
       --  end
   
         cost = 0
         timer = torch.Timer() 
      end

      if math.fmod(d, 1000) == 0 then
         local prevIterParams = util.deepClone(metaLearner.params)   
         torch.save("metaLearner_params_snapshot.th", 
            autograd.util.cast(prevIterParams, "float"))
      end   
      c2=collectgarbage("count")
--      print("what happen end:",c2)
            collectgarbage("collect")
      collectgarbage("collect")
      collectgarbage("collect")
      collectgarbage("collect")
c3 = collectgarbage("count")
--      print("after collect",c3)
   end
   
   ----------------------------------------------------------------------------
   -- meta-testing
   local ret = {} 
   -- number of episodes loop
   _.each(opt.nTest, function(i, n)
      local acc = {}
      for _, k in pairs(opt.nTestShot) do
         acc[k] = torch.zeros(n)
      end
      
      -- episodes loop
 --     print("!!!!!!!!",n)
      for d=1,n do 
	 print("test",d)
         local trainSet, testSet = metaTestSet.createEpisode({})   

         local trainData = trainSet:get() 
         local testData = testSet:get()
         local loss, prediction 

         -- k-shot loop
    --     for _, k in pairs(opt.nTrainShot) do 
    k=opt.nTrainShot
         --   local trainInput, trainTarget = util.extractK(trainData.input, 
         --      trainData.target, k, opt.nClasses.test)
            local trainInput = trainData.input
	    local trainTarget = trainData.target

	    local c=torch.CudaTensor(opt.nClasses.train):fill(0.2)
	--    print(trainTarget)
            local trainInputs, trainTargets, trainWeights = require('model.lstm.trainFirstPuLearner')(opt,trainInput,trainTarget,c,1e-3,0.9,50)
        --    print(trainTargets)
	   
            local loss, prediction = classify(metaLearner.params, learner, 
               trainInputs, trainTargets, trainWeights, testData.input, testData.target, 
               opt.nEpochs[k] or opt.nEpochs[opt.nTrainShot], 
               --opt.batchSize[k] or opt.batchSize[opt.nTrainShot], true)  
               1024,true)

            for i=1,prediction:size(1) do
	       prediction[i][opt.nClasses.train+1]=-100
               testConf[k]:add(prediction[i], testData.target[i]) 
            end

            testConf[k]:updateValids()  
            acc[k][d] = testConf[k].totalValid*100
            testConf[k]:zero()
   --      end
            
      end
         k=opt.nTrainShot
     -- for _,k in pairs(opt.nTrainShot) do 
         print('Test Accuracy (' .. n .. ' episodes, ' .. k .. '-shot)')
         print(acc[k]:mean())
    --  end
 
      ret[n] = _.values(_.map(acc, function(i, val) 
            local low = val:mean() - 1.96*(val:std()/math.sqrt(val:size(1)))
            local high = val:mean() + 1.96*(val:std()/math.sqrt(val:size(1)))       
            return i .. '-shot: ' .. val:mean() .. '; ' .. val:std() 
               .. '; [' .. low .. ',' .. high .. ']' 
      end))
   end)

   return ret
end
