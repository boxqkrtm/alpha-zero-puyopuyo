'''
reused code from

https://github.com/suragnair/alpha-zero-general/blob/master/othello/pytorch/NNet.py

'''


import os
import sys
import time


import numpy as np
from tqdm import tqdm


sys.path.append('../../')

from utils import *


import torch
import torch.optim as optim

import Puyo_NNET

from Duel import Duel



'''utility classes'''

class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


args = dotdict({
    'epochs': 1000000,
    'lr': 0.0001,
    'batch_size': 1,
    'cuda': torch.cuda.is_available(),
})


def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


class NNetWrapper():
    def __init__(self, nnet):

        #pnet: class from Puyo_NNET

        '''
        확인바람
        '''

        self.nnet = nnet

        if torch.cuda.is_available():
            self.nnet.cuda()

    def predict(self, total_field):
        """
        predict the most profitable puyo placement (for possible movements) as policy
        total_field : from class Duel
        """
        # timing
        start = time.time()

        # preparing input
        g_i=total_field.getGameInfo(0)

        puyo_field_factor = total_field.GetFieldInfo(g_i)
        puyo_Outer_field_factor = total_field.GetOuterFieldInfo(g_i)

        puyo_Outer_field_factor =np.hstack(np.array(puyo_Outer_field_factor))

        print(puyo_Outer_field_factor)

        puyo_field_factor=torch.FloatTensor(puyo_field_factor)
        puyo_Outer_field_factor=torch.FloatTensor(puyo_Outer_field_factor)


        if args.cuda:
            puyo_field_factor = puyo_field_factor.contiguous().cuda()
            puyo_Outer_field_factor = puyo_Outer_field_factor.contiguous().cuda()


        puyo_field_factor = puyo_field_factor.view(-1,12)
        puyo_Outer_field_factor.view(-1,14) #필드 이외의 요소들을 전부 받아서 14열로 정리

        ''' not activate drop out option'''
        self.nnet.eval()

        with torch.no_grad():
            pi, v = self.nnet(puyo_field_factor,puyo_Outer_field_factor)

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def train(self, examples):
        """
        each example is of form [f_i,of_i, pi,v, target_pi, target_v]
        f_i: field factor
        of_i: outer field factor
        pi: policy
        v: value(expected winrate of 1P) of this statue
        pi: target policy (evaluated from ISMCTS)
        v: target value (evaluated from simulation)
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr= args.lr)

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            l2_reg = torch.tensor(0.0)

            for param in self.nnet.parameters():
                l2_reg += torch.norm(param)


            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                f_i, of_i, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                f_i = torch.FloatTensor(np.array(boards).astype(np.float64))
                of_i = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if torch.cuda.is_available():
                    f_i, of_i, target_pis, target_vs = f_i.contiguous().cuda(),of_i.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v + 0.1 * l2_reg

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)



                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    '''절대경로로 해야 할 수 있음'''

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
