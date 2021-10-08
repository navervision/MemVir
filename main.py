'''
MemVir
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''

import os
import sys
import glob
import random
import shutil
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter

import net
import loss
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--data', help='path to dataset')
parser.add_argument('--data_name', default=None, type=str,
                    help='dataset name')
parser.add_argument('--save_path', default=None, type=str,
                    help='where your models will be saved')
parser.add_argument('--max_to_keep', default=1, type=int,
                    help='how many keep your saved models')
parser.add_argument('--check_epoch', default=5, type=int,
                    help='do eval every check_epoch')
parser.add_argument('-j', '--workers', default=5, type=int,
                    help='number of data loading workers')
parser.add_argument('--epochs', default=50, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    help='mini-batch size')
parser.add_argument('--modellr', default=0.0001, type=float,
                    help='initial model learning rate')
parser.add_argument('--centerlr', default=0.01, type=float,
                    help='initial center learning rate')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    help='weight decay', dest='weight_decay')
parser.add_argument('--gpu', default=None, type=str,
                    help='GPU id to use.')
parser.add_argument('--eps', default=0.01, type=float,
                    help='epsilon for Adam')
parser.add_argument('--decay_rate', default=0.1, type=float,
                    help='decay rate')
parser.add_argument('--decay_step', default=20, type=int,
                    help='decay step')
parser.add_argument('--decay_stop', default=100000, type=int,
                    help='decay stop')
parser.add_argument('--dim', default=64, type=int,
                    help='dimensionality of embeddings')
parser.add_argument('--freeze_BN', action='store_true',
                    help='freeze bn')
parser.add_argument('-C', default=98, type=int,
                    help='C')
parser.add_argument('--backbone', default='bninception', type=str,
                    help='bninception, resnet18, resnet34, resnet50, resnet101')
parser.add_argument('--pooling_type', default='GAP', type=str,
                    help='GAP | GMP | GAP,GMP')
parser.add_argument('--optimizer', default='adam', type=str,
                    help='adam | adamw')
parser.add_argument('--eval_best', action='store_true',
                    help='eval best saved model')
parser.add_argument('--k_list', default='1,2,4,8', type=str,
                    help='Recall@k list')
parser.add_argument('--input_size', default=224, type=int,
                    help='input size')
## soft max variation
parser.add_argument('--do_nmi', action='store_true',
                    help='do nmi or not')
parser.add_argument('--loss', default='SoftMax_vanilla', type=str,
                    help='loss you want')
parser.add_argument('--scale', default=1.0, type=float,
                    help='scale for softmax variations')
parser.add_argument('--train_with_l2norm', default=True, type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
                    help='use l2norm before criterion')
parser.add_argument('--init_type', default='normal', type=str,
                    help='select normal | uniform for proxy weights')
parser.add_argument('--n_instance', default=1, type=int,
                    help='n_instance')
parser.add_argument('--early_stop_epoch', default=0, type=int,
                    help='early stop if there is no performance increase for such epochs')
parser.add_argument('--use_amp', default=False, type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
                    help='use AMP')
parser.add_argument('--deterministic', default=False, type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
                    help='deterministic experiments')

## arguments for MemVir
parser.add_argument('--memvir', default=False, type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
                    help='Use MemVir training strategy')
parser.add_argument('--warm_epoch', default=0, type=int, help='warm up epoch U_e for MemVir')
parser.add_argument('--mem_num_step', default=-1, type=int, help='number of steps N to use for MemVir')
parser.add_argument('--mem_step_gap', default=1, type=int, help='gap M between steps for MemVir')


def main():
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.data_name.lower() in ["car", "cars", "cars196"]:
        args.C = 98
        args.k_list = '1,2,4,8'
    elif args.data_name.lower() in ["sop", "stanfordonlineproducts"]:
        args.C = 11318
        args.k_list = '1,10,100,1000'
    elif args.data_name.lower() in ["cub", "cub200"]:
        args.C = 100
        args.k_list = '1,2,4,8'
    elif args.data_name.lower() in ['inshop']:
        args.C = 3997
        args.k_list = '1,10,20,40'
    else:
        print("Using custom dataset")

    if args.deterministic:
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        np.random.seed(0)
        torch.backends.cudnn.benchmark = False
        random.seed(0)

    # save training arguments in the save_path
    if args.eval_best:
        args.save_path = os.path.join(args.save_path, 'best')
    if not os.path.exists(args.save_path):
        if args.eval_best:
            print('Train model first!')
            exit()
        os.makedirs(args.save_path)

    args_file = os.path.join(args.save_path, "args.txt")
    with open(args_file, "w") as tf:
        tf.write('\n'.join(sys.argv[1:]))

    # define data loader
    traindir = os.path.join(args.data, 'train')
    testdir = os.path.join(args.data, 'test')

    if 'resnet' in args.backbone:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        scale_value = 1
    else:
        normalize = transforms.Normalize(mean=[104., 117., 128.],
                                         std=[1., 1., 1.])
        scale_value = 255

    train_loader = utils.call_train_loader(traindir, args,
                                            transforms.Compose([
                                               transforms.Lambda(utils.RGB2BGR),
                                               transforms.RandomResizedCrop(args.input_size),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Lambda(lambda x: x.mul(scale_value)),
                                               normalize,
                                            ]))

    test_transforms = transforms.Compose([transforms.Lambda(utils.RGB2BGR),
                                          transforms.Resize(256),
                                          transforms.CenterCrop(args.input_size),
                                          transforms.ToTensor(),
                                          transforms.Lambda(lambda x: x.mul(scale_value)),
                                          normalize,])
    test_image = datasets.ImageFolder(testdir, test_transforms)

    test_class_dict, max_r = utils.get_class_dict(test_image)
    args.test_class_dict = test_class_dict
    args.max_r = max_r

    test_loader = torch.utils.data.DataLoader(
        test_image,
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.data_name.lower() == 'inshop':
        image_info = np.array(test_image.imgs)
        print('\tcheck: gallery == %s, query == %s\n' % (
        image_info[0, 0].split('/')[-3], image_info[-1, 0].split('/')[-3]))
        args.query_labels = np.array(
            [info[0].split('/')[-2] for info in image_info[image_info[:, 1] == '1']])  # 14218 images
        args.gallery_labels = np.array(
            [info[0].split('/')[-2] for info in image_info[image_info[:, 1] == '0']])  # 12612 images
        if len(args.query_labels) != 14218 or len(args.gallery_labels) != 12612:
            print('check you inshop DB')
            exit()

    # Initialize MemVir class
    if args.memvir:
        memvir = loss.MemVir(args)
    else:
        memvir = None

    # define backbone
    if args.backbone == 'bninception':
        model = net.bninception().cuda()
    else:  # resnet family
        model = net.Resnet(resnet_type=args.backbone).cuda()

    # define pooling method
    pooling = net.pooling(pooling_type=args.pooling_type.split(',')).cuda()

    # define embedding method
    embedding = net.embedding(input_dim=model.output_dim, output_dim=args.dim).cuda()

    # define loss function (criterion) and optimizer
    if args.loss.lower() == 'NormSoftmax'.lower():
        criterion = loss.NormSoftmax(args.dim, args.C, scale=args.scale, memvir=memvir).cuda()
    elif args.loss.lower() == 'ProxyNCA'.lower():
        criterion = loss.ProxyNCA(args.dim, args.C, scale=args.scale, init_type=args.init_type, memvir=memvir).cuda()
    else:
        raise ValueError("{} is not supported loss name".format(args.loss))

    params_list = [{"params": model.parameters(), "lr": args.modellr},
                   {"params": embedding.parameters(), "lr": args.modellr},
                   {"params": criterion.parameters(), "lr": args.centerlr}]

    if args.optimizer.lower() == 'Adam'.lower():
        optimizer = torch.optim.Adam(params_list, eps=args.eps, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'AdamW'.lower():
        optimizer = torch.optim.AdamW(params_list, eps=args.eps, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'RMSprop'.lower():
        optimizer = torch.optim.RMSprop(params_list, alpha=0.99, weight_decay=args.weight_decay, momentum=0.9)
    elif args.optimizer.lower() == 'SGD'.lower():
        optimizer = torch.optim.SGD(params_list, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)

    if not args.deterministic:
        cudnn.benchmark = True

    ## do train and test!
    metric_list = ['Recall_1', 'RP', 'MAP']
    best_dict = {'Recall_1': 0.0,
                 'RP': 0.0,
                 'MAP': 0.0}
    best_check = {'Recall_1': False,
                  'RP': False,
                  'MAP': False}
    current_dict = {'Recall_1': 0.0,
                    'RP': 0.0,
                    'MAP': 0.0}

    k_list = [int(k) for k in args.k_list.split(',')]  # [1, 2, 4, 8]
    global_step = 0

    # resume model
    if args.save_path is not None and os.path.exists(args.save_path):
        pth_list = sorted(glob.glob(os.path.join(args.save_path, '*.pth')))
        if len(pth_list) != 0:
            latest_pth = pth_list[-1]
            load_state = torch.load(latest_pth)
            try:
                # for backward compatibility
                best_recall = load_state['best_acc']
                recall_1 = load_state['acc']
                print('\n\n\tResume pretrained models %d epoch %.4f recall_1\n\n' % (load_state['epoch'], recall_1))
            except:
                best_dict = load_state['best_acc']
                current_dict = load_state['acc']
                print('\n\n\tResume pretrained models %d epoch, recall_1, RP, MAP: %.2f, %.2f, %.2f \n\n' % (load_state['epoch'], current_dict['Recall_1'], current_dict['RP'], current_dict['MAP']))
            args.start_epoch = load_state['epoch']  # - 1
            try:     
                global_step = load_state['global_step']
            except:
                global_step = 0

            # state
            model.load_state_dict(load_state['model_state'])
            embedding.load_state_dict(load_state['embedding_state'])
            criterion.load_state_dict(load_state['criterion_state'])
            optimizer.load_state_dict(load_state['optimizer'])


    if not args.eval_best:
        writer = SummaryWriter(args.save_path)
    else:
        args.epochs = 1000000

    if args.use_amp:
        try:
            from torch.cuda.amp import GradScaler, autocast
            scaler = GradScaler()  # Creates a GradScaler for AMP
            print('Running with AMP')
        except:
            args.use_amp = False
            scaler = None
            autocast = None
            print('Failed importing AMP, so just running without AMP')
    else:
        print('Running without AMP')
        scaler = None
        autocast = None

    early_stop_count = 0
    for epoch in range(args.start_epoch, args.epochs):
        epoch += 1
        print('Training in Epoch[{}]'.format(epoch))
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        if not args.eval_best:
            global_step = train(train_loader, model, pooling, embedding, criterion, optimizer, writer, global_step,
                                epoch, memvir, scaler, autocast, args)

        # evaluate on validation set
        if epoch % args.check_epoch == 0:
            nmi, recall, RP, MAP, features, labels = validate(test_loader, model, pooling, embedding, k_list, args)
            print(
                    'Recall@1: {recall[0]:.4f}; RP: {RP:.4f}; MAP: {MAP:.4f} \n'.format(
                    recall=recall, RP=RP, MAP=MAP))

            if args.eval_best:
                print('Evaluation of best saved model is done')
                exit()

            for k_idx, k in enumerate(k_list):
                writer.add_scalar('eval_epoch/Recall_%d' % k, recall[k_idx], epoch)
                writer.add_scalar('eval_step/Recall_%d' % k, recall[k_idx], global_step)
                writer.flush()
            writer.add_scalar('eval_epoch/RP', RP, epoch)
            writer.add_scalar('eval_epoch/MAP', MAP, epoch)
            writer.add_scalar('eval_step/RP', RP, global_step)
            writer.add_scalar('eval_step/MAP', MAP, global_step)
            current_dict['Recall_1'] = recall[0]
            current_dict['RP'] = RP
            current_dict['MAP'] = MAP

            if args.save_path is not None:
                early_stop_count = check_best(recall, k_list, best_dict, best_check, current_dict, metric_list, early_stop_count, writer, epoch, global_step)
                # save first then check early_stop_count
                save_state = {'epoch': epoch,
                              'model_state': model.state_dict(),
                              'embedding_state': embedding.state_dict(),
                              'criterion_state': criterion.state_dict(),
                              'acc': current_dict,
                              'best_acc': best_dict,
                              'optimizer': optimizer.state_dict(),
                              'global_step': global_step}

                save_checkpoint(save_state, best_check, args.max_to_keep, args.save_path, writer)

                if early_stop_count == args.early_stop_epoch and args.early_stop_epoch != 0:
                    print('Exit training due to no performance increase for {} epochs'.format(args.early_stop_epoch))
                    break
            
            best_str = 'Best'
            for metric_name in metric_list:
                best_str += ' %s: %.4f' % (metric_name, best_dict[metric_name])
            print(best_str)
            print('')


def check_best(recall, k_list, best_dict, best_check, current_dict, metric_list, early_stop_count, writer, epoch, global_step):
    '''
    metric_list = ['Recall_1', 'RP', 'MAP']
    writer = Tensorboard wirter
    '''
    for metric_name in metric_list:
        if best_dict[metric_name] < current_dict[metric_name]:
            best_check[metric_name] = True
            best_dict[metric_name] = current_dict[metric_name]
            early_stop_count = -1
            writer.add_scalar('eval_epoch/%s_best' % metric_name, best_dict[metric_name], epoch)
            writer.add_scalar('eval_step/%s_best' % metric_name, best_dict[metric_name], global_step)
            writer.add_scalar('eval_best/%s_best' % metric_name, best_dict[metric_name], epoch)
            for recall_k, k in zip(recall[1:], k_list[1:]):
                writer.add_scalar('eval_best/Recall_%d' % k, recall_k, epoch)
            writer.flush()
        else:
            best_check[metric_name] = False

    return early_stop_count + 1


def swap_idx(array, now_, next_):
    tmp = array[now_]
    array[now_] = array[next_]
    array[next_] = tmp

    return array


def train(train_loader, model, pooling, embedding, criterion, optimizer, writer, global_step, epoch, memvir, scaler,
          autocast, args):
    # switch to train mode
    model.train()
    embedding.train()
    criterion.train()

    if args.freeze_BN:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    total_iter = len(train_loader)
    for i, (input, target) in enumerate(train_loader):

        if args.gpu is not None:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        def forward(input, memvir, target, criterion, args):
            # compute output
            output = model(input)
            output = pooling(output)
            output = embedding(output, l2_norm=args.train_with_l2norm)

            if args.memvir:
                memvir.get_memory(epoch)
                memvir.put_memory(output, target, criterion, epoch)

            loss = criterion(output, target)

            return loss, output

        if args.use_amp:
            with autocast():
                loss, output = forward(input, memvir, target, criterion, args)
        else:
            loss, output = forward(input, memvir, target, criterion, args)

        # TODO: Unify train_info type as dict for every loss
        if i % 10 == 0:
            print('[%d/%d] loss: %.4f' % (i + 1, total_iter, loss.item()))
            writer.add_scalar('train/loss', loss, global_step)
            writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], global_step)
            writer.flush()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        global_step += 1

    return global_step


def validate(test_loader, model, pooling, embedding, k_list, args):
    # switch to evaluation mode
    model.eval()
    embedding.eval()

    testdata = torch.Tensor()
    testdata_l2 = torch.Tensor()
    testlabel = torch.LongTensor()
    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
            if args.gpu is not None:
                input = input.cuda(non_blocking=True)

            # compute output
            output = model(input)
            output = pooling(output)
            output = embedding(output, l2_norm=False)
            output_l2 = F.normalize(output, p=2, dim=1)  # TODO
            testdata = torch.cat((testdata, output.cpu()), 0)
            testdata_l2 = torch.cat((testdata_l2, output_l2.cpu()), 0)
            testlabel = torch.cat((testlabel, target))

    features = testdata.numpy()
    features_l2 = testdata_l2.numpy()
    labels = testlabel.numpy()
    nmi, recall, RP, MAP = utils.evaluation(features_l2, labels, k_list, args)

    return nmi, recall, RP, MAP, features, labels


def adjust_learning_rate(optimizer, epoch, args):
    if epoch % args.decay_step == 0 and epoch <= args.decay_stop:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.decay_rate
            print(param_group['lr'])


def save_checkpoint(state, best_check, max_to_keep, save_path, writer, filename='model.pth'):
    '''
    save_path = args.save_path #folder
    '''
    filename = filename.replace('.pth', '_%05d.pth' % state['epoch'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    pth_save_path = os.path.join(save_path, filename)
    
    torch.save(state, pth_save_path)

    # check max_to_keep
    if max_to_keep != 0:
        for legacy_file in sorted(glob.glob(os.path.join(save_path, '*.pth')))[:-max_to_keep]:
            os.remove(legacy_file)

    # check best_check dict
    for metric_name, is_best in best_check.items():
        best_save_path = os.path.join(save_path, 'best_%s' % metric_name)
        if not os.path.exists(best_save_path):
            os.makedirs(best_save_path)

        if is_best:
            for legacy_file in glob.glob(os.path.join(best_save_path, '*')):
                os.remove(legacy_file)

            pth_best_save_path = os.path.join(best_save_path, filename)
            shutil.copyfile(pth_save_path, pth_best_save_path)


if __name__ == '__main__':
    main()
