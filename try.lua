local _ = require 'moses'
local c = require 'trepl.colorize'
local Dataset = require 'dataset.Dataset'
local trainSet = {}

for i=1,5 do
	_.push(trainSet,{x=i,y=i+1})
end
local aa={}
aa["train"]=1
aa["val"]=5
aa["test"]=4
local split="train"
print(aa[split])
