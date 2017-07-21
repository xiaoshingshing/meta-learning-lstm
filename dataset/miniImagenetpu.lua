local _ = require 'moses'
local c = require 'trepl.colorize'
local Dataset = require 'dataset.Dataset'

local function loadSplit(splitFile)
   --[[
   Args:
      splitFile (string): path to split file
   --]]
   local classes = { }

   local fid = io.open(splitFile, 'r')
   local i = 1
   for line in fid:lines() do
      if i > 1 then
         local parsedLine = string.split(line, ',')
         local filename = parsedLine[1]
         local class = parsedLine[2]
         if not classes[class] then
            classes[class] = { }
         end
         _.push(classes[class], filename)
      end
      i = i + 1
   end
   fid:close()

   return classes
end

local function processor(idx, opt, input)
   nPosClass = opt.nPosClass
   split = opt.split
   if not threadInitialized then
      __ = require 'moses'
      local nn = require 'nn'
                local sys = require 'sys'
      local image = require 'image'

      local pre = nn.Sequential()
      pre:add(nn.Reshape(1, opt.imageDepth, opt.imageHeight, opt.imageWidth))
      pre:add(nn.MulConstant(1.0 / 255, true))

      if opt.train then
         pre:training()
      else
         pre:evaluate()
      end

      if opt.cuda then
         require 'cunn'
         pre:insert(nn.Copy('torch.ByteTensor', 'torch.CudaTensor'), 1)
         pre:cuda()
      else
         pre:insert(nn.Copy('torch.ByteTensor', 'torch.FloatTensor'), 1)
         pre:float()
      end

      function loadImage(path)
         local img = image.load(path, 3, 'byte')
         return image.scale(img, opt.imageWidth, opt.imageHeight)
      end

      function preprocessImages(images)
         return torch.cat(__.map(images, function(i, v)
            return pre:forward(v):clone()
         end), 1)
      end

      threadInitialized = true
   end
   idx = idx % 7 + 1
   opt.classes=__.shuffle(opt.classes)
--   print("!!!!",opt.classes[idx+1],opt.classes[idx+2],opt.classes[idx+3],opt.classes[idx+4],opt.classes[idx+5])
   local trainSet = {}
   local testSet = {}

   local nUnlabel = {}
   nUnlabel["train"]=1
   nUnlabel["val"]=5
   nUnlabel["test"]=4
   local metadata = {}
   local unlabelImages = {}
   -- for 5 positive classes, k labeled samples, 4k unlabeled samples, 15 test samples 
   for i = 1+idx, nPosClass+idx do
      local class = opt.classes[i]
      local urls = __.map(opt.imageFiles[class], function(i, v)
         return opt.dataDir .. '/' .. v
      end)
      urls = __.shuffle(urls)
      local supportUrls = __.first(urls, opt.nSupportExamples*5)
      local evalUrls = __.rest(urls, opt.nSupportExamples*5 + 1)
      local unlabelUrls = __.rest(supportUrls, opt.nSupportExamples + 1)
      supportUrls = __.first(supportUrls, opt.nSupportExamples)
 
      -- filter down evaluation, if necessary
      if opt.nEvalExamples then
         evalUrls = __.first(evalUrls, opt.nEvalExamples)
      end
      local supportImages = {}
      local evalImages =  {}
      for i,url in ipairs(supportUrls) do
	 __.push(supportImages,loadImage(url))
      end
      for i,url in ipairs(unlabelUrls) do
	 __.push(unlabelImages,loadImage(url))
      end
      for i,url in ipairs(evalUrls) do
	 __.push(evalImages,loadImage(url))
      end
      metadata[class]={
	      class = class,
	      supportExamples = preprocessImages(supportImages),
	      evalExamples = preprocessImages(evalImages)
      }
   end
  
   -- for the rest classes, 1(5,4) unlabeled samples for train(val,test)   
   for i= nPosClass+idx+1, #opt.classes do
      local class = opt.classes[i]
      local urls = __.map(opt.imageFiles[class], function(i, v)
         return opt.dataDir .. '/' .. v
      end)
      urls = __.shuffle(urls)
      local unlabelUrls = __.first(urls, nUnlabel[split])
      for i,url in ipairs(unlabelUrls) do
	 __.push(unlabelImages,loadImage(url))
      end
   end
   for i= 1, idx do
      local class = opt.classes[i]
      local urls = __.map(opt.imageFiles[class], function(i, v)
         return opt.dataDir .. '/' .. v
      end)
      urls = __.shuffle(urls)
      local unlabelUrls = __.first(urls, nUnlabel[split])
      for i,url in ipairs(unlabelUrls) do
         __.push(unlabelImages,loadImage(url))
      end
   end

 
   metadata["unlabel"]={
	   class = "unlabel",
	   supportExamples = preprocessImages(unlabelImages),
	   evalExamples = nil
   }
 
   collectgarbage()
   collectgarbage()
 
   return input, metadata
end



local function getData(opt)
   opt.dataCacheDir = opt.dataCacheDir or sys.fpath()
	local imageZipFile = 'images.zip'
	local imagesDir = 'images'
   local splitDir = paths.concat(opt.dataCacheDir, 'miniImagenet')
   local splits = {'train', 'val', 'test'}
   local splitFiles = { }
   _.each(splits, function(i, split)
      splitFiles[split] = split .. '.csv'
   end)
   local requiredFiles = _.append({imageZipFile}, _.values(splitFiles))
	
   if not paths.dirp(splitDir) then
      paths.mkdir(splitDir)
   end
	-- unzip images if necessary
	if not paths.dirp(paths.concat(splitDir, imagesDir)) then 
		print('unzipping: ' .. paths.concat(splitDir, imageZipFile))
		os.execute(string.format('unzip %s -d %s', paths.concat(splitDir, 
         imageZipFile), splitDir))
	end
	
	print('data dir: ' .. paths.concat(splitDir, imagesDir))
	local miniImagenetDataDir = paths.concat(splitDir, imagesDir)

   -- prepare datasets
   local ret = { }
   _.each(splits, function(i, split)
      -- class => image filename mapping
      local imageFiles = loadSplit(paths.concat(splitDir, splitFiles[split]))
      local classes = _.sort(_.keys(imageFiles))
      
      local ds = Dataset(torch.range(1,#classes))
      local get,size = ds.sampledBatcher({
	      inputDims = { 1 },
	      batchSize = opt.nClass[split],
	      samplerKind = opt.episodeSamplerKind,
	      processor = processor,
	      cuda = opt.cuda,
      processorOpt = {
            dataDir = miniImagenetDataDir,
				imageFiles = imageFiles,
            classes = classes,
            nSupportExamples = opt.nSupportExamples,
            nEvalExamples = opt.nEvalExamples,
            classSamplerKind = opt.classSamplerKind,
            imageDepth = opt.imageDepth,
            imageHeight = opt.imageHeight,
            imageWidth = opt.imageWidth,
            resizeType = opt.resizeType,
            pre = opt.pre,
            train = _.contains({'train'}, split),
            cuda = opt.cuda,
            nPosClass = opt.nClass[split],
	    split = split
      }
      })
      ret[split]={
	      get = get,
	      size = size
      }
      end)	
   return ret
end

return getData
