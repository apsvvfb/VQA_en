require 'hdf5'
local utils = require 'misc.utils'
local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)
-- input_img_test_json=opt.input_img_test_json,input_json=opt.input_json,cnn_proto=opt.cnn_proto,cnn_model=opt.cnn_model,vqa_model=vqa_model,batchsize=batch_size
    self.cnnmodel=opt.cnn_model
    self.cnnproto=opt.cnn_proto
    self.vqamodel=opt.vqa_model
    self.batchsize=opt.batch_size
    print('DataLoader loading json file: ', opt.input_img_test_json)
    self.test_file = utils.read_json(opt.input_img_test_json)
    self.idx=1
    self.ques_len_test=#self.test_file

    print('DataLoader loading json file: ', opt.input_json)
    local json_file = utils.read_json(opt.json_file)
    self.ix_to_word = json_file.ix_to_word
    self.ix_to_ans = json_file.ix_to_ans
    self.feature_type = opt.feature_type
    self.seq_length = self.ques_train:size(2)

    -- count the vocabulary key!
    self.vocab_size = utils.count_key(self.ix_to_word)

    collectgarbage() -- do it often and there is no harm ;)
end

function DataLoader:resetIterator(split)
  self.iterators[split] = 1
end

function DataLoader:getVocabSize()
    return self.vocab_size
end

function DataLoader:getSeqLength()
  return self.seq_length
end

function DataLoader:getDataNum(split)
    return #self.split_ix[split]
end

function DataLoader:getBatch(opt)
    local split = utils.getopt(opt, 'split') -- lets require that user passes this in, for safety
    local batch_size = utils.getopt(opt, 'batch_size', 128)

    local split_ix = self.split_ix[split]
    assert(split_ix, 'split ' .. split .. ' not found.')
  
    local max_index = #split_ix
    local infos = {}
    local ques_idx = torch.LongTensor(batch_size):fill(0)
    local img_idx = torch.LongTensor(batch_size):fill(0)

    self.img_batch = torch.Tensor(batch_size, 14, 14, 512)

    for i=1,batch_size do
         local img = self.h5_img_file_train:read('/images_train'):partial({img_idx[i],img_idx[i]},{1,14},
                                    {1,14},{1,512})
         self.img_batch[i] = img
        
    end

    local data = {}
   
    data.images = self.img_batch:view(batch_size, 196, -1):contiguous()
    data.questions = self.ques_test:index(1, ques_idx)
    data.ques_id = self.ques_id_test:index(1, ques_idx)
    data.ques_len = self.ques_len_test:index(1, ques_idx)        
    data.answer = self.ans_test:index(1, ques_idx)        
    
    return data
end
