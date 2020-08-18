# -*- coding: utf-8 -*-
import argparse
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import xml.etree.ElementTree as ET
from flyai.data_helper import DataHelper
from flyai.framework import FlyAI
from path import MODEL_PATH, DATA_PATH
import pandas as pd
from net import get_model

'''
此项目为FlyAI2.0新版本框架，数据读取，评估方式与之前不同
2.0框架不再限制数据如何读取
样例代码仅供参考学习，可以自己修改实现逻辑。
模版项目下载支持 PyTorch、Tensorflow、Keras、MXNET、scikit-learn等机器学习框架
第一次使用请看项目中的：FlyAI2.0竞赛框架使用说明.html
使用FlyAI提供的预训练模型可查看：https://www.flyai.com/models
学习资料可查看文档中心：https://doc.flyai.com/
常见问题：https://doc.flyai.com/question.html
遇到问题不要着急，添加小姐姐微信，扫描项目里面的：FlyAI小助手二维码-小姐姐在线解答您的问题.png
'''
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# 项目的超参，不使用可以删除
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=1, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=2, type=int, help="batch size")
args = parser.parse_args()

# 判断gpu是否可用
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def get_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for object in root.findall('object'):
        # object_name = object.find('label').text
        Xmin = int(object.find('bndbox').find('xmin').text)
        Ymin = int(object.find('bndbox').find('ymin').text)
        Xmax = int(object.find('bndbox').find('xmax').text)
        Ymax = int(object.find('bndbox').find('ymax').text)
        boxes.append([Xmin, Ymin, Xmax, Ymax])
    return boxes

class MyDataset(Dataset):
    def __init__(self, root, img_file_list, xml_file_list, transforms=None):
        self.root = root
        self.transforms = transforms
        self.img_file_list = img_file_list
        self.xml_file_list = xml_file_list

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.img_file_list[idx])
        xml_path = os.path.join(self.root, self.xml_file_list[idx])

        img = Image.open(img_path).convert("RGB")
        boxes = get_xml(xml_path)

        num_objs = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.img_file_list)

def collate_fn(batch):
    return tuple(zip(*batch))


class Main(FlyAI):
    '''
    项目中必须继承FlyAI类，否则线上运行会报错。
    '''

    def download_data(self):
        # 根据数据ID下载训练数据
        data_helper = DataHelper()
        data_helper.download_from_ids("TBDetection")
        print('download data done...')

    def deal_with_data(self):
        '''
        处理数据，没有可不写。
        :return:
        '''
        csv_path = os.path.join(DATA_PATH, 'TBDetection', 'train.csv')
        df = pd.read_csv(csv_path)
        img_file_list = list(df['image_path'].values)
        xml_file_list = list(df['xml_path'].values)
        transform = transforms.Compose([transforms.ToTensor()])
        train_data = MyDataset(os.path.join(DATA_PATH, 'TBDetection'), img_file_list, xml_file_list, transforms=transform)
        self.train_loader = DataLoader(dataset=train_data, batch_size=args.BATCH, shuffle=True, collate_fn=collate_fn)
        self.model = get_model()
        self.model.to(device)
        print('deal with data done...')

    def train(self):
        '''
        训练模型，必须实现此方法
        :return:
        '''
        print('start train...')
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)

        for epoch in range(1, args.EPOCHS+1):
            self.model.train()
            for i, (images, targets) in enumerate(self.train_loader):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                print('epoch: %d, batch: %d, loss: %f'%(epoch, i, losses))
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

            lr_scheduler.step(epoch=epoch)
            torch.save(self.model.state_dict(), os.path.join(MODEL_PATH, 'best.pth'))


if __name__ == '__main__':
    main = Main()
    main.download_data()
    main.deal_with_data()
    main.train()