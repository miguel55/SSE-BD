import itertools

import numpy as np
import torch
import torch.utils.data as data
from torch.autograd import Variable

from cirtorch.datasets.datahelpers import default_loader
from cirtorch.datasets.genericdataset import DataFromMatrix

class CNICTestDataset (data.Dataset):
    """Data loader that processes the dynamic features from CNIC dataset

    Args:
        db (matrix): dynamic feature dataset (NxTxF) with N number of samples, T number of time instants, F number of features
        name (string): dataset name: 'cnic'
        mode (string): 'train' or 'val' for training and validation parts of dataset
        arch (string): architecture to use: 'lstm'
        group (int matrix, Default: None): groups of interest
        n_features (int): Number of features
        output_dim (int): Dimension of the embedding
        pnum (int, Default:5): Number of positives for a query image in a training tuple
        nnum (int, Default:5): Number of negatives for a query image in a training tuple
        qsize (int, Default:1000): Number of query images, ie number of (q,p,n1,...nN) tuples, to be processed in one epoch
        poolsize (int, Default:10000): Pool size for negative images re-mining
        transform(Default: None): Transformations for the input data

     Attributes:
        images (list): List of full filenames for each image
        clusters (list): List of clusterID per image
        qpool (list): List of all query image indexes
        ppool (list): List of positive image indexes, each corresponding to query at the same position in qpool

        qidxs (list): List of qsize query image indexes to be processed in an epoch
        pidxs (list): List of qsize positive image indexes, each corresponding to query at the same position in qidxs
        nidxs (list): List of qsize tuples of negative images
                        Each nidxs tuple contains nnum images corresponding to query image at the same position in qidxs

        Lists qidxs, pidxs, nidxs are refreshed by calling the ``create_epoch_tuples()`` method, 
            ie new q-p pairs are picked and negative images are remined
    """

    def __init__(self, db, name, mode, groups, n_features, outputdim, pnum=1, nnum=5, qsize=5000, poolsize=20000, transform=None, loader=default_loader):

        if not (mode == 'train' or mode == 'val'):
            raise(RuntimeError("MODE should be either train or val, passed as string"))

        ids=db[:,:,n_features:]
        db=db[:,:,:n_features]
        db=db[np.in1d(ids[:,int(db.shape[1]/2),0],np.array(groups)),:,:]
        ids=ids[np.in1d(ids[:,int(db.shape[1]/2),0],np.array(groups)),:,:]

        # initializing tuples dataset
        self.db=db.astype('float32')
        self.name = name
        self.outputdim=outputdim
        self.mode = mode
        self.clusters = ids[:,int(db.shape[1]/2),0]
        self.qpool = range(db.shape[0])
        self.ppool = range(db.shape[0]) 

        # size of training subset for an epoch
        self.pnum = pnum
        self.nnum = nnum
        self.qsize = min(qsize, len(self.qpool))
        self.poolsize = min(poolsize, db.shape[0])
        self.qidxs = None
        self.pidxs = None
        self.nidxs = None

        self.avgPosDist = None
        self.avgNegDist = None
        
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        
        """
        Args:
            index (int): Index

        Returns:
            images tuple (q,p,n1,...,nN): Loaded train/val tuple at index of self.qidxs
        """
        if self.__len__() == 0:
            raise(RuntimeError("List qidxs is empty. Run ``dataset.create_epoch_tuples(net)`` method to create subset for train/val!"))

        output = []
        # query image
        output.append(self.db[index,:,:])
        
        if self.transform is not None:
            output = [self.transform(output[i]) for i in range(len(output))]

        target = torch.Tensor([self.clusters[index]])

        return output, target

    def __len__(self):
        return self.db.shape[0]

    def __repr__(self):
        fmt_str = self.__class__.__name__ + '\n'
        fmt_str += '    Name and mode: {} {}\n'.format(self.name, self.mode)
        fmt_str += '    Number of images: {}\n'.format(len(self.images))
        fmt_str += '    Number of training tuples: {}\n'.format(len(self.qpool))
        fmt_str += '    Number of negatives per tuple: {}\n'.format(self.nnum)
        fmt_str += '    Number of tuples processed in an epoch: {}\n'.format(self.qsize)
        fmt_str += '    Pool size for negative remining: {}\n'.format(self.poolsize)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
    def create_epoch_tuples(self,net, numEpoch=None):
        
        if (self.pnum == 1):
            self.create_epoch_tuples_single_positive(net, numEpoch)
        elif (self.pnum == 0):
            self.create_epoch_tuples_test(net, numEpoch)
        else:
            self.create_epoch_tuples_multiple_positives(net)
        return

    def create_epoch_tuples_single_positive(self, net, numEpoch, print_freq=1000):

        print('>> Creating tuples for an epoch of {}-{}...'.format(self.name, self.mode))

        ## ------------------------
        ## SELECTING POSITIVE PAIRS
        ## ------------------------
    
    
        # draw qsize random queries for tuples
        idxs2qpool = range(len(self.qpool))[:self.qsize]
        qsel = [self.qpool[i] for i in idxs2qpool]
        self.qidxs = []
        self.pidxs = []
        
        # find all positives to each query
        for q in qsel:
            
            pos = []
            for i,x in enumerate(self.qpool):
                if ((self.clusters[q] == self.clusters[x]) and (not q==x)):
                    pos.append(self.ppool[i])
                    
            # select pnum positives
            self.qidxs.append(q)
            self.pidxs.append([pos[i] for i in torch.randperm(len(pos))[:self.pnum]])
            
        ## ------------------------
        ## SELECTING NEGATIVE PAIRS
        ## ------------------------
    
        # if nnum = 0 create dummy nidxs
        # useful when only positives used for training
        if self.nnum == 0:
            self.nidxs = [[] for _ in range(len(self.qidxs))]
            return
    
        # pick poolsize negatives too
        idxs2images = torch.randperm(self.db.shape[0])[:self.poolsize]
            
        # prepare network
        net.cuda()
        net.eval()
        
        print('>> Extracting descriptors for all the data...')
        loader = torch.utils.data.DataLoader(
            DataFromMatrix(self.db, transform=self.transform),
            batch_size=1, shuffle=False, num_workers=8, pin_memory=True
        )
        avecs = torch.Tensor(self.outputdim, self.db.shape[0]).cuda()
        for i, input_ in enumerate(loader):
            if (i+1) % print_freq == 0:
                print('\r>>>> {}/{} done...'.format(i+1, self.db.shape[0]), end='')
            avecs[:, i] = (net(Variable(input_.cuda())).data).squeeze()
        print('')
    
        print('>> Extracting descriptors for query images...')
        qvecs = torch.Tensor(self.outputdim, len(self.qidxs)).cuda()
        
        for i in range(len(self.qidxs)):
            if (i+1) % print_freq == 0:
                print('\r>>>> {}/{} done...'.format(i+1, len(self.qidxs)), end='')
            qvecs[:, i] = avecs[:,self.qidxs[i]]
        print('')
        
        print('>> Extracting descriptors for the first positive image...')
        pvecs = torch.Tensor(self.outputdim, len(self.pidxs)).cuda()
        for i in range(len(self.pidxs)):
            if (i+1) % print_freq == 0:
                print('\r>>>> {}/{} done...'.format(i+1, len(self.pidxs)), end='')
            pvecs[:, i] = avecs[:,self.pidxs[i][0]]
        print('')
    
        print('>> Extracting descriptors for negative pool...')
        poolvecs = torch.Tensor(self.outputdim, len(idxs2images)).cuda()
        for i in range(len(idxs2images)):
            if (i+1) % print_freq == 0:
                print('\r>>>> {}/{} done...'.format(i+1, len(idxs2images)), end='')
            poolvecs[:, i] = avecs[:,idxs2images[i]]
        print('')
        
        ##############################
    
        print('>> Searching for hard negatives...')
        #scores = torch.mm(poolvecs.t(), qvecs)
        #scores, ranks = torch.sort(scores, dim=0, descending=True)
        scores, ranks = mmByParts(poolvecs, qvecs, self.qsize)
        self.nidxs = []
        for q in range(len(self.qidxs)):
            qcluster = self.clusters[self.qidxs[q]]
            clusters = [qcluster]
            nidxs = []
            r = 0
            avg_ndist = torch.Tensor([0]).cuda()
            n_ndist = torch.Tensor([0]).cuda()
            while len(nidxs) < self.nnum:
                potential = idxs2images[ranks[r, q]]
                # take at most one image from the same cluster
                if not self.clusters[potential] in clusters and potential not in nidxs:
                    nidxs.append(potential)
                    clusters.append(self.clusters[potential])
                    avg_ndist += torch.pow(qvecs[:,q]-poolvecs[:,ranks[r, q]]+1e-6, 2).sum(dim=0).sqrt()
                    n_ndist += 1
                r += 1
            self.nidxs.append(nidxs)
            
        # Finally, update the average pos and negative dist
        dif = qvecs - pvecs
        D = torch.pow(dif+1e-6, 2).sum(dim=0).sqrt()
        self.avgPosDist = D.mean()
        self.avgNegDist = (avg_ndist/n_ndist).cpu()[0]
        print('>>>> Average positive distance: {:.4f}'.format(self.avgPosDist))
        print('>>>> Average negative distance: {:.4f}'.format(self.avgNegDist))
        print('>>>> Done')
        return
    
    def create_epoch_tuples_multiple_positives(self,net, print_freq=1000):
        
        print('>> Creating tuples for an epoch of {}-{}...'.format(self.name, self.mode))

        ## ------------------------
        ## SELECTING POSITIVE PAIRS
        ## ------------------------
    
        # draw qsize random queries for tuples
        idxs2qpool = range(len(self.qpool))[:self.qsize]
        qsel = [self.qpool[i] for i in idxs2qpool]
        self.pidxs = []
     
        # find all positives to each query
        for q in qsel:
            
            pos = []
            for i,x in enumerate(self.qpool):
                if ((self.clusters[q] == self.clusters[x]) and (not q==x)):
                    pos.append(self.ppool[i])
                    
            # select pnum positives
            self.pidxs.append([q]+[pos[i] for i in torch.randperm(len(pos))[:self.pnum-1]])
            
        
        # now, lets unroll the list of lists in idx into a single list to find 
        # a hard negative for each positive. We need to keep the info about the
        # positives grouping
        lengths = [len(i) for i in self.pidxs]
        epidxs = list(itertools.chain(*self.pidxs))
        
        # pick poolsize negatives too
        idxs2images = torch.randperm(self.db.shape[0])[:self.poolsize]
            
        # prepare network
        net.cuda()
        net.eval()
    
        print('>> Extracting descriptors for all the data...')
        loader = torch.utils.data.DataLoader(
            DataFromMatrix(self.db, transform=self.transform),
            batch_size=1, shuffle=False, num_workers=8, pin_memory=True
        )
        avecs = torch.Tensor(self.outputdim, self.db.shape[0]).cuda()
        for i, input_ in enumerate(loader):
            if (i+1) % print_freq == 0:
                print('\r>>>> {}/{} done...'.format(i+1, self.db.shape[0]), end='')
            avecs[:, i] = (net(Variable(input_.cuda())).data).squeeze()
        print('')
        
        print('>> Obtaining descriptors for positive pool...')
        qvecs = torch.Tensor(self.outputdim, len(epidxs)).cuda()
        for i in range(len(epidxs)):
            if (i+1) % print_freq == 0:
                print('\r>>>> {}/{} done...'.format(i+1, len(epidxs)), end='')
            qvecs[:, i] = avecs[:,epidxs[i]]
        print('')
    
        print('>> Obtaining descriptors for negative pool...')
        poolvecs = torch.Tensor(self.outputdim, len(idxs2images)).cuda()
        for i in range(len(idxs2images)):
            if (i+1) % print_freq == 0:
                print('\r>>>> {}/{} done...'.format(i+1, len(idxs2images)), end='')
                poolvecs[:, i] = avecs[:,idxs2images[i]]

        print('')
        
        
        print('>> Searching for hard negatives...')
        scores, ranks = mmByParts(poolvecs, qvecs, self.qsize)
        
        self.nidxs = []
        numIm = 0
        
        for group in self.pidxs:
            npos = len(group)
            queries = [ numIm+i for i in range(npos) ]
            
            nidxs = []
            avg_ndist = torch.Tensor([0]).cuda()
            n_ndist = torch.Tensor([0]).cuda()
            clusters = [self.clusters[epidxs[q]] for q in queries] 
            
            for q in queries:
                added = 0
                r = 0
                
                while added == 0:
                    potential = idxs2images[ranks[r, q]]
                    if not self.clusters[potential] in clusters and potential not in nidxs:
                        nidxs.append(potential)
                        added += 1
                        avg_ndist += torch.pow(qvecs[:,q]-poolvecs[:,ranks[r, q]]+1e-6, 2).sum(dim=0).sqrt()
                        n_ndist += 1
                    r += 1
            self.nidxs.append(nidxs)
            numIm = numIm + npos
        
        
    
        # for compatibility with other code, lets label the first positive as query
        self.qidxs = []
        for l in self.pidxs:
            self.qidxs.append(l.pop(0))
        
        # print some info
        self.avgPosDist = -1.0
        self.avgNegDist = (avg_ndist/n_ndist).cpu()[0]
        print('>>>> Average positive distance: {}'.format('Not computed'))
        print('>>>> Average negative distance: {:.4f}'.format(self.avgNegDist))
        print('>>>> Done')
        return
    
    def create_epoch_tuples_test(self, net, numEpoch, print_freq=1000):

        print('>> Creating tuples for an epoch of {}-{}...'.format(self.name, self.mode))

        ## ------------------------
        ## SELECTING POSITIVE PAIRS
        ## ------------------------
    
    
        # draw qsize random queries for tuples
        idxs2qpool = range(len(self.qpool))[:self.qsize]
        qsel = [self.qpool[i] for i in idxs2qpool]
        self.qidxs = []
        
        # find all positives to each query
        for q in qsel:

            # select pnum positives
            self.qidxs.append(q)
            
        ## ------------------------
        ## SELECTING NEGATIVE PAIRS
        ## ------------------------
    
        # if nnum = 0 create dummy nidxs
        # useful when only positives used for training
        if self.nnum == 0:
            self.nidxs = [[] for _ in range(len(self.qidxs))]
            return
            
        # prepare network
        net.cuda()
        net.eval()
        
        print('>> Extracting descriptors for all the data...')
        loader = torch.utils.data.DataLoader(
            DataFromMatrix(self.db, transform=self.transform),
            batch_size=1, shuffle=False, num_workers=8, pin_memory=True
        )
        avecs = torch.Tensor(self.outputdim, self.db.shape[0]).cuda()
        for i, input_ in enumerate(loader):
            if (i+1) % print_freq == 0:
                print('\r>>>> {}/{} done...'.format(i+1, self.db.shape[0]), end='')
            avecs[:, i] = (net(Variable(input_.cuda())).data).squeeze()
        print('')
    
        print('>> Extracting descriptors for query images...')
        qvecs = torch.Tensor(self.outputdim, len(self.qidxs)).cuda()
        
        for i in range(len(self.qidxs)):
            if (i+1) % print_freq == 0:
                print('\r>>>> {}/{} done...'.format(i+1, len(self.qidxs)), end='')
            qvecs[:, i] = avecs[:,self.qidxs[i]]
        print('')
        
        return
    
    def renestList( self, extendedList, lengths):
        
        rebuild = []
        start = 0
        for l in lengths:
            rebuild.append(extendedList[start:l+start])
            start += l
        
        return rebuild
    
def mmByParts( poolvecs, qvecs, qsize, maxRank=1000 ):
    
    step = 1000 if qsize > 1000 else qsize
    start = 0
    end = int(qvecs.size()[1])
    scoresR = torch.cuda.FloatTensor(maxRank, end)
    ranksR = torch.cuda.LongTensor(maxRank, end)
    
    for i in range(int(np.ceil(end/step))):
        
        e = start+step
        e = e if e < end else end
        #print('[:,{}:{}] '.format(start, e))
        
        # compute scores and ranks for the portion of queries
        scores_i = torch.mm(poolvecs.t(), qvecs[:,start:e])
        scores_i, ranks_i = torch.sort(scores_i, dim=0, descending=True)
        
        # store the top maxRank results
        scoresR[:,start:e] = scores_i[0:maxRank]
        ranksR[:,start:e] = ranks_i[0:maxRank]
        
        # update the start point
        start = e
        
    return scoresR, ranksR