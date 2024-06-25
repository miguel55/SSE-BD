import os
import shutil
import time
import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms


# setting path
sys.path.append('../..')
from cirtorch.layers.loss import MyAggregateExponentialGammaLoss, MyAggregateExponentialGammaLossID
from cirtorch.datasets.traindataset import CNICDataset
from cirtorch.datasets.testdataset import CNICTestDataset

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from collections import OrderedDict
import scipy.io as sio


linestyles = OrderedDict(
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])

markers = Line2D.markers

class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def set_batchnorm_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        # freeze running mean and std
        m.eval()

def setRandomSeeds( s ):
    
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    return
 
    
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, lstm_size, output_size, time_length,bidirectional):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        if (bidirectional):
            self.lstm_layer = nn.LSTM(input_size,int(lstm_size/2),1,batch_first=True, bidirectional=True)
        else:
            self.lstm_layer = nn.LSTM(input_size,lstm_size,1,batch_first=True, bidirectional=False)
        self.layer_norm=nn.LayerNorm(lstm_size)
        self.linear=nn.Linear(lstm_size,output_size,bias=False)
        
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x,h=self.lstm_layer(x)
        x=self.layer_norm(x)
        x=self.linear(x[:,int((x.size(1)-1)/2),...].squeeze(1))
        return x

def loadModelToCuda( eConfig ):
    
    # Define my model
    if( eConfig['arch'] == 'lstm' ):
        model = LSTMModel(eConfig['n-features'], eConfig['non-supervised-dim'], eConfig['output-dim'] ,eConfig['time-length'],eConfig['bidirectional'])
    return model.cuda()

def defineLossFunction( eConfig ):

    if eConfig['loss'] == 'myAggExpGamma':
        criterion = MyAggregateExponentialGammaLoss(alpha=eConfig['exp-loss-alpha'],
                                                    gamma=eConfig['gamma-BD'],
                                                    drop_loss=eConfig['drop-loss'], 
                                                    drop_loss_freq=eConfig['drop-loss-freq']).cuda()
        
    elif eConfig['loss'] == 'myAggExpGammaID':
        criterion1 = MyAggregateExponentialGammaLoss(alpha=eConfig['exp-loss-alpha'],
                                                     gamma=eConfig['gamma-BD'], 
                                                     drop_loss=eConfig['drop-loss'], 
                                                     drop_loss_freq=eConfig['drop-loss-freq']).cuda()
        criterion2 = MyAggregateExponentialGammaLossID(alpha=eConfig['exp-loss-alpha-traj'], 
                                                       gamma=eConfig['gamma-TC'],
                                                       drop_loss=eConfig['drop-loss'], 
                                                       drop_loss_freq=eConfig['drop-loss-freq']).cuda()
        criterion=(criterion1,criterion2)
    else:
        raise(RuntimeError("Loss {} not available!".format(eConfig['loss'])))
    
    return criterion

def defineOptimizer( eConfig, model ):
    
    # parameters split into features and pool (no weight decay for pooling layer)
    parameters = [
        {'params': model.parameters(),'lr': eConfig['lr']}    ]

    # define optimizer
    if eConfig['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(parameters, eConfig['lr'], momentum=eConfig['momentum'], weight_decay=eConfig['weight-decay'])
        
    elif eConfig['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(parameters, eConfig['lr'], weight_decay=eConfig['weight-decay'])
        
    else:
        raise(RuntimeError("Optimizer {} not available!".format(eConfig['optimizer'])))
        
    return optimizer

def createDataLoading( eConfig, model ):
    
    if ( eConfig['training-set'] == 'cnic'):
        return createDataLoadingDynFeatures(eConfig, model)
    else:
        pass
    return

def collate_tuples2(batch):
    return [batch[0][0]], [batch[0][1]]

def collate_tuples4(batch):
    if len(batch) == 1:
        return [batch[0][0]], [batch[0][1]], [batch[0][2]], [batch[0][3]]
    return [batch[i][0] for i in range(len(batch))], [batch[i][1] for i in range(len(batch))], [batch[i][2] for i in range(len(batch))], [batch[i][3] for i in range(len(batch))]

def createDataLoadingDynFeatures( eConfig, model ):

    db=sio.loadmat(os.path.join(eConfig['data-dir'],'cnic_dataset.mat'))['features_lstm']
    mean=sio.loadmat(os.path.join(eConfig['data-dir'],'mean_features.mat'))['mean_lstm']
    std=sio.loadmat(os.path.join(eConfig['data-dir'],'std_features.mat'))['std_lstm']
    trajectories_id=sio.loadmat(os.path.join(eConfig['data-dir'],'neutrophils_by_trajectory.mat'))['neutrophils_by_trajectory']
    timestamps=sio.loadmat(os.path.join(eConfig['data-dir'],'neutrophils_by_timestamp.mat'))['neutrophils_by_timestamp']
    normalize = transforms.Normalize(mean=mean, std=std)
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = CNICDataset(
        db=db,
        trajectories_id=trajectories_id,
        timestamps=timestamps,
        name=eConfig['training-set'],
        mode='train',
        arch=eConfig['arch'],
        groups=eConfig['training-groups'],
        n_features=eConfig['n-features'],
        outputdim=eConfig['output-dim'],
        pnum = eConfig['pos-num'],
        nnum = eConfig['neg-num'],
        pnum_traj = eConfig['pos-num-traj'],
        nnum_traj = eConfig['neg-num-traj'],
        qsize= eConfig['query-size'],
        poolsize = eConfig['pool-size'],
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=eConfig['batch-size'], shuffle=False,
        num_workers=eConfig['workers'], pin_memory=False, sampler=None,
        drop_last=True, collate_fn=collate_tuples4
    )

    val_dataset = CNICDataset(
        db=db,
        trajectories_id=trajectories_id,
        timestamps=timestamps,
        name=eConfig['training-set'],
        mode='val',
        arch=eConfig['arch'],
        groups=eConfig['test-groups'],
        n_features=eConfig['n-features'],
        outputdim=eConfig['output-dim'],
        pnum = eConfig['pos-num'],
        nnum = eConfig['neg-num'],
        pnum_traj = eConfig['pos-num-traj'],
        nnum_traj = eConfig['neg-num-traj'],
        qsize= float('Inf'),
        poolsize = float('Inf'),
        transform=transform
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=eConfig['batch-size'], shuffle=False,
        num_workers=eConfig['workers'], pin_memory=False,
        drop_last=True, collate_fn=collate_tuples4
    )
    
    return train_dataset, train_loader, val_dataset, val_loader

def createTestLoaderDynFeatures (eConfig, model):

    db=sio.loadmat(os.path.join(eConfig['data-dir'],'cnic_dataset.mat'))['features_lstm']
    mean=sio.loadmat(os.path.join(eConfig['data-dir'],'mean_features.mat'))['mean_lstm']
    std=sio.loadmat(os.path.join(eConfig['data-dir'],'std_features.mat'))['std_lstm']
    normalize = transforms.Normalize(mean=mean, std=std)
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    test_dataset = CNICTestDataset(
        db=db,
        name=eConfig['training-set'],
        mode='val',
        groups=eConfig['test-groups'],
        n_features=eConfig['n-features'],
        outputdim=eConfig['output-dim'],
        pnum = 0,
        nnum = 0,
        qsize= float('Inf'),
        poolsize = float('Inf'),
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=eConfig['workers'], pin_memory=False,
        drop_last=True, collate_fn=collate_tuples2
    )
    
    return test_dataset, test_loader


def train(eConfig, train_loader, model, criterion, optimizer, epoch):
        
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    criterion1, criterion2 = criterion

    # create tuples for training
    train_loader.dataset.create_epoch_tuples(model)    

    # switch to train mode
    model.train()
    model.apply(set_batchnorm_eval)

    end = time.time()
    optimizer.zero_grad()
    for i, (input1_, target1, input2_, target2) in enumerate(train_loader):
    
        # measure data loading time
        data_time.update(time.time() - end)
        nq = len(input1_) # number of training tuples

        for q in range(nq):
            
            ni1 = len(input1_[q]) # number of images in qth tuple
            ni2 = len(input2_[q]) # number of images in qth tuple
            output1 = torch.Tensor(eConfig['output-dim'], ni1).cuda()
            output2 = torch.Tensor(eConfig['output-dim'], ni2).cuda()

            target_var1 = target1[q].cuda()
            target_var2 = target2[q].cuda()

            for imi in range(ni1):
                output1[:,imi] = model(input1_[q][imi].cuda()).squeeze()
                
            for imi in range(ni2):
                output2[:,imi] = model(input2_[q][imi].cuda()).squeeze()
            
            loss1 = criterion1(output1, target_var1, train_loader.dataset.avgPosDist, train_loader.dataset.avgNegDist)#, Lw=model.meta['Lw'])
            if (torch.sum(target_var2==1)>0):
                loss2 = criterion2(output2, target_var2, train_loader.dataset.avgPosDist_traj, train_loader.dataset.avgNegDist_traj)#, Lw=model.meta['Lw'])
                loss = loss1+loss2
            else: 
                loss=loss1
            losses.update(loss.item())
            loss.backward()
            
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % eConfig['print-freq'] == 0:
            print('>> Train: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch+1, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
    
    return losses.avg

def validate(eConfig, val_loader, model, criterion, epoch):
    
    batch_time = AverageMeter()
    losses = AverageMeter()

    # create tuples for validation
    val_loader.dataset.create_epoch_tuples(model)

    # switch to evaluate mode
    model.eval()
    criterion1, criterion2 = criterion

    end = time.time()
    for i, (input1_, target1, input2_, target2) in enumerate(val_loader):

        nq = len(input1_) # number of training tuples
        ni1 = len(input1_[0]) # number of images per tuple
        ni2 = len(input2_[0]) # number of images per tuple
        output1 = torch.autograd.Variable(torch.Tensor(eConfig['output-dim'], nq*ni1).cuda(), volatile=True)
        output2 = torch.autograd.Variable(torch.Tensor(eConfig['output-dim'], nq*ni2).cuda(), volatile=True)

        for q in range(nq):
            for imi in range(ni1):
                input_var = torch.autograd.Variable(input1_[q][imi].cuda())

                # compute output
                output1[:, q*ni1 + imi] = model(input_var)
                
            for imi in range(ni2):
                input_var = torch.autograd.Variable(input2_[q][imi].cuda())

                # compute output
                output2[:, q*ni2 + imi] = model(input_var)

        target_var = torch.autograd.Variable(torch.cat(target1).cuda())
        loss1 = criterion1(output1, target_var)

        target_var = torch.autograd.Variable(torch.cat(target2).cuda())
        loss2 = criterion1(output2, target_var)

        loss=loss1+loss2
        # record loss
        losses.update(loss.item()/nq, nq)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % eConfig['print-freq'] == 0:
            print('>> Val: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch+1, i, len(val_loader), batch_time=batch_time, loss=losses))

    return losses.avg

def test(eConfig, model_path, epoch):
    
    
    print('>> Evaluating network on test datasets...')
    
    setRandomSeeds(0)

    # load initial model to cuda
    model = loadModelToCuda( eConfig )

    # load weights 
    model.load_state_dict(torch.load(os.path.join(eConfig['result-dir'],model_path,'model_epoch'+str(epoch)+'.pth.tar'))['state_dict'])

    # moving network to gpu and eval mode
    model.cuda()
    model.eval()
    
    test_dataset, test_loader = createTestLoaderDynFeatures( eConfig, model )

    # create tuples for validation
    test_loader.dataset.create_epoch_tuples(model)

    output = np.zeros((eConfig['output-dim'], len(test_loader)),dtype='float32')
    target = np.zeros((1, len(test_loader)),dtype='float32')
    
    for i, (input_, t) in enumerate(test_loader):
        # target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input_[0][0].cuda())

        # compute output
        output[:,i] = model(input_var).detach().cpu().numpy()
        
        # save target
        target[:,i] = t[0].detach().cpu().numpy()

        print('%d out of %d' %(i,len(test_loader)))

    embedding=np.concatenate((output,target),axis=0)
    
    return embedding


def trainAndVal( eConfig, saveResults=False):

    setRandomSeeds(0)

    # load initial model to cuda
    model = loadModelToCuda( eConfig )

    # create save dir
    if saveResults:
        saveDIR = os.path.join(eConfig['result-dir'],eConfig['arch']+'_g_{}_gtraj_{}_lr_{}_bs_{}_pnum_{}_traj_{}_out_{}'.format(eConfig['gamma-BD'],eConfig['gamma-TC'],
            eConfig['lr'], eConfig['batch-size'], eConfig['pos-num'], eConfig['pos-num-traj'], eConfig['output-dim']))
        print(">> Creating directory if it does not exist:\n>> '{}'".format(saveDIR))
        if not os.path.exists(saveDIR):
            os.makedirs(saveDIR)
            
    # define loss function, optimizer and lr schedule
    criterion = defineLossFunction( eConfig )
    optimizer = defineOptimizer( eConfig, model)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=math.exp(eConfig['lr-expDecay']))
    
    # create dataset and loader
    train_dataset, train_loader, val_dataset, val_loader = createDataLoading( eConfig, model)


    epoch_loss=np.zeros(eConfig['epochs'],dtype=float)

    if (not os.path.exists(os.path.join(saveDIR,'model_epoch%d.pth.tar' % 60))):

        for epoch in range(eConfig['epochs']):
    
            # set manual seeds per epoch
            setRandomSeeds(epoch)
    
            # adjust learning rate for each epoch
            scheduler.step()
            lr_feat = optimizer.param_groups[0]['lr']
            print('>> Features lr: {:.2e}'.format(lr_feat))
                
            # train for one epoch on train set
            lossTr = train( eConfig, train_loader, model, criterion, optimizer, epoch)
            
            # evaluate on validation set
            if eConfig['run-validation']:
                epoch_loss[epoch] = validate(eConfig, val_loader, model, criterion, epoch)
            else:
                epoch_loss[epoch] = lossTr
    
            # evaluate on test datasets
            if (epoch+1) % eConfig['save-interval'] == 0 or (epoch+1) == eConfig['epochs']:
                if saveResults:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'min_loss': epoch_loss[epoch],
                        'optimizer' : optimizer.state_dict(),
                    }, False, saveDIR)
                    
                    fig1 = plt.gcf()
                    plt.plot(epoch_loss[0:epoch+1])
                    plt.ylabel('Loss')
                    fig1.savefig(os.path.join(saveDIR,'Loss.png'))
                    plt.close('all')  
                    outLogFile=os.path.join(saveDIR, 'loss.txt')
                    np.savetxt(outLogFile,np.vstack((range(epoch+1),epoch_loss[0:epoch+1])).transpose(),"%d %.3f")    
        
        
def testModel(modelPath, eConfig):
    
    # load original model
    model = loadModelToCuda( eConfig )
    
    # load the learned weigths
    checkpoint = torch.load(modelPath)
    model.load_state_dict(checkpoint['state_dict'])
            
    # test the model
    r = test(model, eConfig)
    return r

def save_checkpoint(state, is_best, directory):
    filename = os.path.join(directory, 'model_epoch%d.pth.tar' % state['epoch'])
    torch.save(state, filename)
    if is_best:
        filename_best = os.path.join(directory, 'model_best.pth.tar')
        shutil.copyfile(filename, filename_best)

if __name__ == '__main__':
           
    torch.cuda.device(0)
    cd = torch.cuda.current_device()
    print('count: {}'.format(torch.cuda.device_count()))
    print('current_device: {}'.format(cd))
    print('device name: {}'.format(torch.cuda.get_device_name(cd)))
      
    eConfig = {
        'result-dir': 'results/train',
        'data-dir': 'data',
        'training-set': 'cnic',
        'arch':'lstm',
        'training-groups': [1,2], # groups_for_training
        'test-groups': [1,2,3,4], # groups_for_test
        'n-features': 21, # number of input features
        'time-length': 21, # number of input features
        'non-supervised-dim': 256, # model output dim
        'output-dim': 16, # model output dim
        'bidirectional':True,
        'run-validation': False,
        'loss': 'myAggExpGammaID', 
        'drop-loss': 0,
        'drop-loss-freq': 5,
        'gamma-BD': 0.8, # gamma_BD parameter
        'exp-loss-alpha': 1.0, # for the margin ratio in our Bag Exponential 
        'gamma-TC': 0.1,  # gamma_TC parameter
        'exp-loss-alpha-traj': 1.0, # for the margin ratio in our Bag Exponential 
        'optimizer': 'adam', # adam
        'lr': 2.5e-6,
        'lr-expDecay': -0.01,
        'momentum': 0.9, #0.9
        'weight-decay': 1e-4,#1e-4 seems to work
        'epochs': 60,
        'batch-size': 40, # the number of tuples (q,p1,p2..,n1,n2,...) per batch
        'accumulate': 20, # the number of batches to accum for one param update
        'image-size': 1024, #362
        'neg-num': 30,  # the number of negatives in a training tuple
        'pos-num': 30, # the number of positives in a training tuple
        'neg-num-traj': 11,  # the number of negatives in a training tuple (par)
        'pos-num-traj': 11, # the number of positives in a training tuple (par)
        'query-size': float('Inf'), # the number of tuples per epoch
        'pool-size': float('Inf'), # the negative pool size 
        'workers': 0,
        'print-freq': 20,
        'save-interval': 1,
        'crop-scale': 1.0}
    
    args = sys.argv[1::]
    for i in range(0,len(args),2):
        key = args[i]
        val = args[i+1]
        eConfig[key] = type(eConfig[key])(val)
        print (str(eConfig[key]))
          
    print('args')
    print(sys.argv)
    print('eConfig')
    print(eConfig)
    torch.multiprocessing.set_sharing_strategy('file_system')
    # train
    trainAndVal(eConfig, saveResults=True)
    
    
    model_path=eConfig['arch']+'_g_{}_gtraj_{}_lr_{}_bs_{}_pnum_{}_traj_{}_out_{}'.format(eConfig['gamma-BD'],eConfig['gamma-TC'],eConfig['lr'],eConfig['batch-size'],eConfig['pos-num'],eConfig['pos-num-traj'],eConfig['output-dim'])
    if (os.path.exists(os.path.join(eConfig['result-dir'],model_path,'loss.txt'))):
        loss=np.loadtxt(os.path.join(eConfig['result-dir'],model_path,'loss.txt'))#"%d %.3f"    
        epoch=np.argmin(loss[np.where(np.isnan(loss[:,1])==False),1])+1
        embedding=test(eConfig,model_path,epoch)
        sio.savemat(os.path.join(eConfig['result-dir'],model_path,'embedding_best.mat'),{'embedding': embedding})
