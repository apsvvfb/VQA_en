import sys
import json 
import numpy as np
filename="/home/c-nrong/VQA/draw/Json/question_answers_genome.json"
trainIDs="/home/c-nrong/VQA/Jan_HieCoAttenVQA/genIDs/trainID.txt"
testIDs="/home/c-nrong/VQA/Jan_HieCoAttenVQA/genIDs/testID.txt"
maxNum=8
outfile2="noids.txt"
jfile = json.load(open(filename, 'r'))
imdir='%s/%s.jpg'
out=[]

train="VG_100K"
test="VG_100K_2"

f=open(trainIDs,'r')
trainline=f.readlines()
f.close()
trainlines=map(str.strip, trainline)

f=open(testIDs,'r')
testline=f.readlines()
f.close()
testlines=map(str.strip, testline)

for i in range(len(jfile)):
    image_id = str(jfile[i]['id'])
    questions = jfile[i]['qas']
    NumQues = len(questions)
    for qid in range(0,NumQues):
	ans = questions[qid]['answer'] 
        question = questions[qid]['question']
        question_id = questions[qid]['qa_id']
	if qid < maxNum :
	    if image_id in trainlines:
	        image_path = imdir%(train, image_id)
                out.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'ans': ans})
	    elif image_id in testlines:
		image_path = imdir%(test, image_id)
	        out.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'ans': ans})
	    else:
		with open(outfile2,'a') as f:
		    f.write(image_id+"\n")
print("there are %d samples!" % len(out))
json.dump(out, open('vqa_raw_test.json', 'w')) 

