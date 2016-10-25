require 'os'
split_test=torch.Tensor({{2,2,2,2,2,2,2,2}})
split_ix={}
iterators={}
for i = 1,split_test:size(1) do 
	idx = split_test[i] 
	if not split_ix[idx] then  
		split_ix[idx] = {} 
		iterators[idx] = 1 
	end 
	--os.exit()
	table.insert(split_ix[idx], i) 
end 
print(split_ix)
print(iterators)
