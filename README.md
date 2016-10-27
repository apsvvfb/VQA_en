[path: /home/c-nrong/VQA/HieCoAttenVQA2/eval.lua]

input: question_answers_genome.json (used as testset) + trainID+testID + images
output: data/vqa_data_prepro.json(ix_to_img_test) + 'AttenmapsAndImgidx.h5('attenmaps','imgidx')
trainset: VQA dataset
testset: the dataset we wanna evaluation (genome)
#generate vqa_raw_train.json +  vqa_raw_test.json
  cd /home/c-nrong/VQA/HieCoAttenVQA2/
  cd data/
  python vqa_preprocess.py --download 1 --split 1
  rm vqa_raw_test.json
  python vqa_preprocess_Eval.py
#download cnn model
  cd image_model/
  python download_model.py --download 'VGG'
#Generate Question Features (data/vqa_data_prepro.h5, data/vqa_data_prepro.json)
  cd  prepro/
  python prepro_vqa.py --input_train_json ../data/vqa_raw_train.json --input_test_json ../data/vqa_raw_test.json --num_ans 1000
#generate image feature(vqa_data_img_vgg_train.h5)
(we don't need to extract feature for training set,so I comment some lines of prepro_img_vgg.lua)
  cd prepro/
  th prepro_img_vgg.lua -input_json ../data/vqa_data_prepro.json -image_root /home/c-nrong/VQA/ -cnn_proto ../image_model/VGG_ILSVRC_19_layers_deploy.prototxt -cnn_model ../image_model/VGG_ILSVRC_19_layers.caffemodel
#eval.lua(AttenmapsAndImgidx.h5)
  rm AttenmapsAndImgidx.h5
  th eval.lua

