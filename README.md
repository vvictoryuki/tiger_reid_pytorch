## The solution for ICCV 2019 Workshop CVWC (Plain ReID Track)

### Introduction 
&ensp;&ensp; For our solution, we extract global features and local features of images on CNN, optimize these features with Triplet loss and id classification loss and apply several methods for data augmentation such as flip as new id, random whitening, random crop and so on. Besides, we proposed an example sampling strategy for training using hard negative mining. Finally, we ensemble our models with different backbones and epochs using imagenet pre-trained models (ResNet50, ResNet152, DenseNet161, DenseNet169, DenseNet201, DenseNet121) downloaded from pytorch.The challege result can be found at https://cvwc2019.github.io/leaderboard.html.

### Dependencies
- python = 2.7.16
- pytorch = 0.4.1
- numpy = 1.11.3
- opencv-python = 3.2.0.7
- scipy = 0.18.1

### Prepare Data
You can download the data from https://cvwc2019.github.io/challenge.html.

- create the image list
```
cd tiger_dataset
python split.py
```
In folder tiger_dataset, we have already prepared the image list in all_list.txt for you. It is noted that not each image has its id (only images of the tigers' right/left sides have been annotated). In reid_list_train.txt we list all ids and the corresponding images. By running split.py you can get train.txt and test.txt in which we list the train set information and validation set information.

- prepare data for train&test
```
cd ..
mkdir images
python partition.py
```
By runnng partition.py you can create the images folder containing all prepared images and the partition.pkl which is used to tell the train.py which image is belong to the train set and which image is belong to the test set (query and gallery information is also included).

### Trian&Test
```
python train.py --total_epochs 30 --exp_decay_at_epoch 15
```
The default setting: DenseNet-161, global loss:local loss:id classification loss = 0.6:0:0.3 .
If you want to use specific GPUs, you can use '-d' or '--sys_device_ids'. For example, if you want to use GPU1 and GPU2, just use:
```
train train.py -d (1,2)
```
If you want to apply some data augmentation methods to the dataset, you can modify the transform.py. If you want to use other models to train, you can modify the tiger_reid_pytorch/aligned_reid/model/Model.py.
After training, this code will run on validation set automatically and output the accuracy result (mAP, top-1, top-5, top-10).

##### If you any questions, please contact vvictoryuki@163.com.



