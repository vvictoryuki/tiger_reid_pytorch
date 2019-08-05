import os,pdb
import numpy as np


f_1 = open('no_label.txt','w')

known_names = []
with open('reid_list_train.txt','r') as f:
   lines = f.readlines()
   for line in lines:
     id = int(line.split(',')[0])
     name = line.split(',')[1].split('.')[0]+'.jpg'
     known_names.append(name)

with open('all_list.txt','r') as f:
   lines = f.readlines()
   for line in lines:
      name = line.split('.')[0]+'.jpg'
      if name not in known_names:
         f_1.write('-1,'+name+'\n')
        
f_1.close()
