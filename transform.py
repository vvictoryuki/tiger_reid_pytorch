# encoding: utf-8
import shutil
import os.path as osp
import cv2
from aligned_reid.utils.utils import save_pickle
if __name__ == '__main__':
    train_im_names = []
    test_im_names = []
    test_marks = []
    new_im_dir = './images'
    im_path = './tiger_dataset/atrw_reid_train/train'
    save_path = './'
    train_path = './tiger_dateset/train.txt'
    test_path = './tiger_dateset/test.txt'

    # train_ids_num = 0
    with open(train_path, 'r') as f:
        lines = f.readlines()
        ids = []
        last_id = -100
        cam = 0
        for line in lines:
            id = int(line.split(',')[0])
            if id == -1:
                break
            elif id != last_id:
                cam = 0
                ids += [id, ]
                ids += [id+300,]
                # train_ids_num += 1
            else:
                cam += 1
            last_id = id
            name = line.split(',')[1].split('.')[0] + '.jpg'
            ori_name = name.split('.')[0]
            new_im_name = '{:08d}_{:04d}_{:08d}.jpg'.format(id, cam, int(ori_name))
            #new_im_name_2 = '{:08d}_{:04d}_{:08d}.jpg'.format(id, cam+300, int(ori_name))
            #new_im_name_flip = '{:08d}_{:04d}_{:08d}.jpg'.format(id+300, cam, int(ori_name))
            shutil.copy(osp.join(im_path, name), osp.join(new_im_dir, new_im_name))

            
            #ori_img = cv2.imread(osp.join(im_path , name))
            #flip_img = cv2.flip(ori_img, 1)
            #cv2.imwrite(osp.join(new_im_dir , new_im_name_flip), flip_img)
            #cv2.imwrite(osp.join(new_im_dir , new_im_name_2), flip_img)
            
            train_im_names.append(new_im_name)
            #train_im_names.append(new_im_name_2)
            #train_im_names.append(new_im_name_flip)
        train_ids2labels = dict(zip(ids, range(len(ids))))
        # print(len(ids))
       
    with open(test_path, 'r') as f:
        lines = f.readlines()
        last_id = -100
        cam = 0
        for line in lines:
            id = int(line.split(',')[0])
            if id != last_id:
                cam = 0
            else:
                cam += 1
            last_id = id
            name = line.split(',')[1]#.split('.')[0] + '.jpg'
            ori_name = name.split('.')[0]
            marks = int(line.split(',')[2].split()[0])
            new_im_name = '{:08d}_{:04d}_{:08d}.jpg'.format(id, cam, int(ori_name))
            shutil.copy(osp.join(im_path, name), osp.join(new_im_dir, new_im_name))
            test_im_names.append(new_im_name)
            test_marks.append(marks)
    # print(len(train_im_names),len(train_ids2labels))
    partitions = {'train_im_names': train_im_names,
                  'train_ids2labels': train_ids2labels,
                  'test_im_names': test_im_names,
                  'test_marks': test_marks}
    partition_file = osp.join(save_path, 'partitions.pkl')
    save_pickle(partitions, partition_file)
    print('Partition file saved to {}'.format(partition_file))

