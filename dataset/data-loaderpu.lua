local _ = require 'moses'

function check(rawData, opt, k)
   if #rawData.item <= 0 or (opt.nClasses[k] 
      and #rawData.item < opt.nClasses[k]) then
      
      return false
   end  

   return true
end

return function(opt) 
   require('dataset.dataset')

   local dataOpt = {
      cuda = opt.useCUDA,
      episodeSamplerKind = opt.episodeSamplerKind or 'permutation',  

      dataCacheDir = opt.rawDataDir,

      nClass = opt.nClasses,
      nSupportExamples = math.max(opt.nTrainShot, math.max(unpack(opt.nTestShot))),
      nEvalExamples = opt.nEval,

      imageDepth = opt.nDepth,
      imageHeight = opt.nIn,
      imageWidth = opt.nIn,
      resizeType = 'scale',
      
      normalizeData = opt.normalizeData
   }
  
   local data = require('dataset.miniImagenetpu')(dataOpt)
--   local data = require(opt.dataName)(dataOpt) 
   
   local function prepareDataset(split, sample, field, batchSize)
      local examples=nil
      local classes=nil
      local num = 1
--      sample.item=_.shuffle(sample.item)
      for i, url in pairs(sample.item[1].extra) do
--	 print(i)
	 if(i=='unlabel') then
            if(url[field]) then
               if examples==nil then
	          examples = url[field]
	          classes = url[field].new(url[field]:size(1)):fill(opt.nClasses.train+1)
	       else
                  examples = torch.cat(examples,url[field],1)
	          classes = torch.cat(classes,url[field].new(url[field]:size(1)):fill(opt.nClasses.train+1),1)
	       end
            end
	 else
            if examples==nil then
               examples = url[field]
	       classes = url[field].new(url[field]:size(1)):fill(num)
	    else
	    --   print(num)
	      -- print(examples)
	       examples = torch.cat(examples, url[field], 1)
               classes = torch.cat(classes, url[field].new(url[field]:size(1)):fill(num),1)
            end
	    num = num + 1
         end
--	 print("!!!!!!!!!!!!!")
--	 print(i)
--	 print(url)
--         if(num == 1) then
--            examples = url[field]
--	    classes = url[field].new(url[field]:size(1)):fill(num)
--         else
--	    if(url[field]) then
--	       examples = torch.cat(examples, url[field], 1)
--	       classes = torch.cat(classes, url[field].new(url[field]:size(1)):fill(num),1)
--	    end
--         end
--	 num = num + 1
      end
    
      local ds = Dataset({ x = examples, y = classes, batchSize=batchSize, shuffle=true})

      return ds 
   end

   _.each(data, function(k,v)
--      print("?????",data[k].get)
      if type(data[k]) == 'table' and data[k].get then
         data[k].createEpisode = 
            function(lopt)
               local rawData = data[k].get()
               while not check(rawData, opt, k) do
                  rawdata = data[k].get()
               end
               local trainDataset, testDataset = prepareDataset(k, 
                  rawData, 'supportExamples', lopt.trainBatchSize), 
                  prepareDataset(k, rawData, 'evalExamples', lopt.testBatchSize)
                              
               return trainDataset, testDataset
            end 
      end 
   end)  

   return data.train, data.val, data.test
end
