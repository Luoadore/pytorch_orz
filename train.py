# -*- encoding=utf-8 -*- 
import sys
from PIL import Image
import os,random,time
import pickle
import torch
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import *

import torch.nn.functional as F
from loader import mura_data
from torchvision.models import densenet169

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
tf_train = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.Resize((440,440)),
#         transforms.CenterCrop((400,400)),
        transforms.RandomCrop((400,400)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
tf_test = transforms.Compose([
#         transforms.CenterCrop((400,400)),
        transforms.Resize((400,400)),
        transforms.ToTensor(),
        normalize,
    ])
device = torch.device("cuda",1)
def main(opt):
    train_dataset = mura_data(lists =opt['train_path'],
                     transform = tf_train)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                            batch_size=opt['batch_size'], shuffle=True,num_workers=3)

    test_dataset = mura_data(lists =opt['test_path'],
                     train = False,     
                     transform = tf_test)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                            batch_size=opt['batch_size'], shuffle=False,num_workers=3)
    
    model = densenet169(pretrained='/root/workspace/pre_models/densenet169.pth')
    model.fc = torch.nn.Linear(1664,1)
    model.to(device)
    
    optimizer = torch.optim.Adam(
            model.parameters(),lr=opt['lr'], weight_decay=1e-3)
#     optimizer = torch.optim.SGD(
#             model.parameters(),lr=opt['lr'], weight_decay=1e-4)
    binary_loss_fn = torch.nn.BCEWithLogitsLoss(weight=None)
    
    start_epoch = 1
    results = {'loss':10,'kappa':0,'train':[],'test':[],'loss_list':[],'kappa_list':[]}
    for epoch in range(start_epoch, opt['num_epochs']):
        #utils.normalize_batch(x)
        if opt['decay_epochs']:
            adjust_learning_rate(optimizer, epoch, initial_lr=opt['lr'], decay_epochs=opt['decay_epochs'])
        train_metrics = train_epoch(epoch, model, train_loader, optimizer, binary_loss_fn)
        eval_metrics,out_list = val(model, test_loader, binary_loss_fn,opt['test_path'])
        results['train'].append(train_metrics)
        results['test'].append(eval_metrics)
        
        if eval_metrics[1]<results['loss']:
            results['loss'] = eval_metrics[1]
            results['loss_list'] = out_list
            if epoch>1:
                model.cpu()
                torch.save(model.state_dict(), opt['out_dir']+'.th') 
                print('save to %s'%opt['out_dir'])
                model.to(device)
                
        if eval_metrics[-2]>results['kappa']:
            results['kappa'] = eval_metrics[-2]
            results['kappa_list'] = out_list
            if epoch>1:
                model.cpu()
                torch.save(model.state_dict(), opt['out_dir']+'_kappa.th') 
                print('save to %s'%opt['out_dir'])
                model.to(device)
        x = open(opt['out_dir']+'.pkl','wb')
        pickle.dump(results,x)
        x.close()
        
def train_epoch(epoch, model, loader, optimizer, binary_loss_fn):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    prec1_m = AverageMeter()
#     type_acc_m = AverageMeter()
    model.train()
    end = time.time()
    for batch_idx,(input, target, type_label) in enumerate(loader):
        input, target, type_label = input.to(device), target.to(device), type_label.to(device)
        b_output = model(input)
        loss = binary_loss_fn(b_output, target)#+type_loss_fn(t_output,type_var.squeeze_())

        losses_m.update(loss.item(), input.size(0))
        prec1= accuracy(b_output, target)
        prec1_m.update(prec1, b_output.size(0))
#         type_acc= accuracy2(t_output.data, type_label)
#         type_acc_m.update(type_acc[0], t_output.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_time_m.update(time.time() - end)
        end = time.time()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]  '
                  'acc:{acc.val:.4f} ({acc.avg:.4f}) '
                  'Loss: {loss.val:.6f} ({loss.avg:.4f})  '
                  'Time: {batch_time.val:.3f}s, {rate:.3f}/s  '.format(
                epoch,
                batch_idx, len(loader),
                100.*batch_idx / len(loader),
                loss=losses_m,
                batch_time=batch_time_m,
                rate=input.size(0) / batch_time_m.val,
                acc=prec1_m))
#                 type_acc=type_acc_m.val[0],
#                 type_acc_avg=type_acc_m.avg[0]))
    return [prec1_m.avg,losses_m.avg]

def val(model, loader, binary_loss_fn,val_list):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    prec1_m = AverageMeter()
#     type_acc_m = AverageMeter()
    with open(val_list) as f:
        eval_list = [x.strip() for x in f.readlines()]
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        output_list = []
        for i,(input, target,type_label) in enumerate(loader):
            input, target, type_label = input.to(device), target.to(device), type_label.to(device)
            b_output = model(input)
            loss = binary_loss_fn(b_output, target)
            losses_m.update(loss.item(), input.size(0))
            prec1= accuracy(b_output, target)
            prec1_m.update(prec1, b_output.size(0))
            pos = F.sigmoid(b_output)
            output_list += pos.data.view(-1).tolist()
            batch_time_m.update(time.time() - end)
            end = time.time()

            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                  'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                  'Prec@1 {acc.val:.4f} ({acc.avg:.4f})'.format(
                i, len(loader),
                batch_time=batch_time_m, loss=losses_m,
                acc=prec1_m))
    auc,kappa1 = kappa(output_list,eval_list)
    print('AUC:%f'%auc)
    print('kappa:',kappa1[0],kappa1[1])

    return [prec1_m.avg,losses_m.avg,auc]+kappa1,output_list


def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        thred = torch.ge(output,0.0)
        correct = torch.eq(thred.float(),target)
        acc = correct.float().sum(0).mul_(100.0 / batch_size)
        return acc.item()
def adjust_learning_rate(optimizer, epoch, initial_lr, decay_epochs=30):
    lr = initial_lr * (0.9 ** (epoch // decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def kappa(outputs,eval_list):
    auc = roc_auc_score([int('positive' in p) for p in eval_list],outputs)
    results = {}
    for i,p in enumerate(eval_list):
        k = p.split('/')[2]+p.split('/')[3]+ p.split('/')[4]
        if k not in results:
            results[k] = [int('positive' in p)]
        results[k].append(outputs[i])

    total = [[],[],[]]
    for k in results:
        total[0].append(results[k][0])
        score1 = max(results[k][1:])
        score2 = sum(results[k][1:])/len(results[k][1:])
        total[1].append(int(score1>0.5))
        total[2].append(int(score2>0.5))

    max_kappa = cohen_kappa_score(total[0],total[1])
    sum_kappa = cohen_kappa_score(total[0],total[2])
    return auc,[sum_kappa,max_kappa]
if __name__ == '__main__':
    opt = {}
    opt['train_path'] = '/root/workspace/data/MURA-v1.1/train_image_paths.csv'
    opt['test_path'] = '/root/workspace/data/MURA-v1.1/valid_image_paths.csv'
    opt['num_classes'] = 7
    opt['lr'] = 5e-5
    opt['pre_lr'] = 1e-5
    opt['drop_rate']=0.5
    opt['num_epochs'] = 100
    opt['decay_epochs'] = 4#False#1
    opt['batch_size'] = 16
    opt['pre_model'] = False#'models/model_1.th'
    opt['out_dir'] = 'parameters/dense_base'
    main(opt)