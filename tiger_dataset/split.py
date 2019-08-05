import os,pdb
import numpy as np

ids = {}
with open('reid_list_train.txt','r') as f:
   lines = f.readlines()
   for line in lines:
      id = int(line.split(',')[0])
      name = line.split(',')[1].split('.')[0]+'.jpg'
      if id in ids:
        ids[id].append(name)        
      else:
        ids[id] = [name]
        
f_train = open('train.txt','w')
f_test = open('test.txt','w')

train_id_num = 0
test_id_num = 0
train_im_num = 0
test_im_num = 0
for id in ids:
   if np.random.random()<0.53:
      # train
      train_id_num += 1
      for name in ids[id]:
         f_train.write(str(id)+','+name+'\n')
         train_im_num += 1
   else:
      # test
      test_id_num += 1
      np.random.shuffle(ids[id])
      images_num = len(ids[id])
      query_num = max(int(images_num*0.33),1)
      print(images_num)
      for index,name in enumerate(ids[id]):
        test_im_num += 1
        if index<query_num:
          f_test.write(str(id)+','+name+',0\n')
        else:
          f_test.write(str(id)+','+name+',1\n')

print('train_id_num:'+str(train_id_num)+' test_id_num:'+str(test_id_num))
print('train_im_num:'+str(train_im_num)+' test_im_num:'+str(test_im_num))
f_train.close()
f_test.close()
