import argparse
import sys
import os
import warnings
import gc
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchcontrib.optim import SWA
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from sampnn.message import CompositionNet
from sampnn.data import input_parser, CompositionData 
from sampnn.data import Normalizer
from sampnn.data import collate_batch
from sampnn.utils import train, evaluate, save_checkpoint


args = input_parser()

def main():
    global args, best_error

    device = torch.device("cuda") if args.cuda else torch.device("cpu")

    # load data
    dataset = CompositionData(args.data_dir, seed=42)

    params = {  'batch_size': args.batch_size,
                'num_workers': args.workers, 
                'pin_memory': False,
                'shuffle':False,
                'collate_fn': collate_batch}

    total = len(dataset)
    indices = list(range(total))
    train_idx = int(total * args.train_size) # note int() truncates but this same as floor for +ve 
    val_idx = int(total * args.val_size) + train_idx
    test_idx = total - train_idx - val_idx

    train_set = torch.utils.data.Subset(dataset, indices[:train_idx])
    val_set = torch.utils.data.Subset(dataset, indices[train_idx:val_idx])
    test_set = torch.utils.data.Subset(dataset, indices[-test_idx:])

    # val_set = torch.utils.data.Subset(dataset, indices[train_idx:])
    # test_set = copy.deepcopy(val_set)

    train_generator = DataLoader(train_set, **params)
    val_generator = DataLoader(val_set, **params)
    test_generator = DataLoader(test_set, **params)

    # for large data sets we can use a subset for the normaliser
    _, sample_target, _ = collate_batch(train_set)
    normalizer = Normalizer(sample_target)

    # build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[1].shape[-1]
    nbr_fea_len = structures[2].shape[-1]

    output_nn = nn.Sequential(nn.Linear(args.atom_fea_len,96), nn.BatchNorm1d(96),
                          nn.ELU(),nn.Linear(96,48), nn.BatchNorm1d(48), 
                          nn.ELU(), nn.Linear(48,1))

    # output_nn = nn.Sequential(nn.Linear(args.atom_fea_len,96), nn.BatchNorm1d(96),
    #                       nn.ELU(),nn.Linear(96,1))

    crys_gate = nn.Sequential(nn.Linear(args.atom_fea_len, 96), nn.BatchNorm1d(96),
                            nn.Softplus(), nn.Linear(96,1))

    pool_gate = nn.Sequential(nn.Linear(args.atom_fea_len, 96),nn.BatchNorm1d(96),
                                nn.Softplus(),nn.Linear(96,1))

    # crys_gate = nn.Sequential(nn.Linear(args.atom_fea_len, 1))

    # pool_gate = nn.Sequential(nn.Linear(args.atom_fea_len, 1))
                
    model = CompositionNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=args.atom_fea_len,
                                n_graph=args.n_conv,
                                output_nn=output_nn,
                                crys_gate=crys_gate,
                                atom_gate=pool_gate)

    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total Number of Trainable Parameters: {}'.format(num_param))

    writer = SummaryWriter()
    # val_writer = SummaryWriter()
    
    if args.cuda:
        model.cuda()

    # Loss Function
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()

    # args.optim = 'Adam'

    # Choose Optimiser
    if args.optim == 'SGD':
        base_optim = optim.SGD(model.parameters(), args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        base_optim = optim.Adam(model.parameters(), args.learning_rate,
                               weight_decay=args.weight_decay)
    else:
        raise NameError('Only SGD or Adam is allowed as --optim')

    optimizer = SWA(base_optim, swa_start=10, swa_freq=5, swa_lr=0.05)
    # optimizer = base_optim

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_error = checkpoint['best_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        _, best_error = evaluate(val_generator, model, criterion, normalizer,
                        verbose=False)

    # decay the learning rate multiplicatively by gamma every time a 
    # milestone is reached.
    # scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,
    #                         gamma=0.4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=50)
    
    try:
        for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
            # Training
            model.train()
            train_loss, train_error = train(train_generator, model, criterion, 
                                            optimizer, normalizer, args.cuda)

            # Validation
            with torch.set_grad_enabled(False):
                # switch to evaluate mode
                model.eval()
                # evaluate on validation set
                val_loss, val_error = evaluate(val_generator, model, criterion, 
                                                normalizer, args.cuda)

            print('Epoch: [{0}/{1}]\t'
                    'Train : Loss {2:.4f}\t'
                    'Error {3:.3f}\t'
                    'Validation : Loss {4:.4f}\t'
                    'Error {5:.3f}\n'.format(
                    epoch+1, args.start_epoch + args.epochs, train_loss, train_error,
                    val_loss, val_error))

            scheduler.step()

            is_best = val_error < best_error
            if is_best:
                best_error = val_error

            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_error': best_error,
                    'optimizer': optimizer.state_dict(),
                    'normalizer': normalizer.state_dict(),
                    'args': vars(args)
                }, is_best)


            writer.add_scalar('data/train', train_error, epoch+1)
            writer.add_scalar('data/validation', val_error, epoch+1)

            if epoch % 25 == 0:
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch+1)
                    # writer.add_histogram(name+'/grad', param.grad.clone().cpu().data.numpy(), epoch+1)

            # catch memory leak
            gc.collect()

    except KeyboardInterrupt:
        pass


    # test best model
    # note this returns the model with the best average MSE this can
    # be skewed heavily by the fact that a single minibatch in the
    # epoch had an uncharacteristically low MSE.
    print('---------Evaluate Model on Test Set---------------')
    # best_checkpoint = torch.load('checkpoint.pth.tar')
    best_checkpoint = torch.load('model_best.pth.tar')
    print(best_checkpoint['epoch'])
    model.load_state_dict(best_checkpoint['state_dict'])
    evaluate(test_generator, model, criterion, normalizer, test=True)






if __name__ == '__main__':
    main()
