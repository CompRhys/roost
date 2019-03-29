import argparse
import sys
import os
import csv
import shutil
import warnings
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.functional import l1_loss as mae
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from sampnn.message import CompositionNet
from sampnn.data import input_parser, CompositionData 
from sampnn.data import collate_batch
from sampnn.data import AverageMeter, Normalizer


args = input_parser()

def main():
    global args, best_mae_error

    device = torch.device("cuda") if args.cuda else torch.device("cpu")

    # load data
    dataset = CompositionData(*args.data_options)

    params = {  'batch_size': args.batch_size,
                'num_workers': args.workers, 
                'pin_memory': False,
                'shuffle':True,
                'collate_fn': collate_batch}

    total = len(dataset)
    indices = list(range(total))
    train_idx = int(total * args.train_size) # note int() truncates but this same as floor for +ve 
    val_idx = int(total * args.val_size)
    test_idx = total - train_idx - val_idx

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_idx, val_idx, test_idx])

    train_generator = DataLoader(train_set, **params)
    val_generator = DataLoader(val_set, **params)
    test_generator = DataLoader(test_set, **params)

    
    # for large data sets we can use a subset for the normaliser
    _, sample_target, _ = collate_batch(dataset)
    normalizer = Normalizer(sample_target)

    # build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CompositionNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=args.atom_fea_len,
                                n_graph=args.n_conv,
                                h_fea_list=args.h_fea_list)
    if args.cuda:
        model.cuda()

    # Loss Function
    criterion = nn.MSELoss()

    # Choose Optimiser
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.learning_rate,
                               weight_decay=args.weight_decay)
    else:
        raise NameError('Only SGD or Adam is allowed as --optim')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mae_error = checkpoint['best_mae_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # decay the learning rate multiplicatively by gamma every time a 
    # milestone is reached.
    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,
                            gamma=0.1)

    best_mae_error = evaluate(val_generator, model, criterion, normalizer,
                                verbose=False)
    

    train_losses = []
    validation_losses = []

    for epoch in range(args.start_epoch, args.epochs):
        
        # Training
        model.train()
        train(train_generator, model, criterion, optimizer, epoch, normalizer)

        # Validation
        with torch.set_grad_enabled(False):
            # switch to evaluate mode
            model.eval()
            # evaluate on validation set
            mae_error = evaluate(val_generator, model, criterion, normalizer)

            if mae_error != mae_error:
                print('Exit due to NaN')
                sys.exit(1)

        scheduler.step()

        # remember the best mae_eror and save checkpoint
        is_best = mae_error < best_mae_error
        best_mae_error = min(mae_error, best_mae_error)

        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_mae_error': best_mae_error,
                'optimizer': optimizer.state_dict(),
                'normalizer': normalizer.state_dict(),
                'args': vars(args)
            }, is_best)

    # test best model
    print('---------Evaluate Model on Test Set---------------')
    best_checkpoint = torch.load('model_best.pth.tar')
    model.load_state_dict(best_checkpoint['state_dict'])
    evaluate(test_generator, model, criterion, normalizer, test=True)


def train(train_loader, model, criterion, optimizer, 
            epoch, normalizer, verbose = False):
    """
    run a forward pass, backwards pass and then update weights
    """
    losses = AverageMeter()
    mae_errors = AverageMeter()

    for i, (input_, target, _) in enumerate(train_loader):
        
        # normalize target
        target_var = normalizer.norm(target)
        
        if args.cuda:
            input_ = (input_[0].cuda(async=True),
                        input_[1].cuda(async=True),
                        input_[2].cuda(async=True),
                        input_[3].cuda(async=True),
                        input_[4].cuda(async=True),
                        input_[5].cuda(async=True))
            target_var = target_var.cuda(async=True)

        # compute output
        output = model(*input_)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu().item(), target.size(0))
        mae_errors.update(mae_error, target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose:
            print('Batch: [{0}/{1}]\t'
                    'Loss {loss.val:.4f}\t'
                    'MAE {mae_errors.val:.3f}'.format(
                    i, len(train_loader), loss=losses, mae_errors=mae_errors))

    print('Epoch: [{0}/{1}]\t'
        'Loss {loss.avg:.4f}\t'
        'MAE {mae_errors.avg:.3f}'.format(
        epoch, args.epochs, loss=losses, mae_errors=mae_errors))

    pass
    

def evaluate(generator, model, criterion, normalizer, 
                test=False, verbose=False):
    """ evaluate the model """
    losses = AverageMeter()
    mae_errors = AverageMeter()

    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    if test:
       label = 'Test'
    else:
        label = 'Validate'

    for i, (input_, target, batch_cif_ids) in enumerate(generator):
        
        # normalize target
        target_var = normalizer.norm(target)
        
        if args.cuda:
            input_ = (input_[0].cuda(async=True),
                        input_[1].cuda(async=True),
                        input_[2].cuda(async=True),
                        input_[3].cuda(async=True),
                        input_[4].cuda(async=True),
                        input_[5].cuda(async=True))
            target_var = target_normed.cuda(async=True)

        # compute output
        output = model(*input_)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu().item(), target.size(0))
        mae_errors.update(mae_error, target.size(0))

        if verbose:
            print('{0}: [{1}/{2}]\t'
                    'Loss {loss.val:.4f}\t'
                    'MAE {mae_errors.val:.3f}'.format(label,
                    i, len(train_loader), loss=losses, mae_errors=mae_errors))

        if test:
            test_pred = normalizer.denorm(output.data.cpu())
            test_target = target
            test_preds += test_pred.view(-1).tolist()
            test_targets += test_target.view(-1).tolist()
            test_cif_ids += batch_cif_ids

    print('{0}: \t'
        'Loss {loss.avg:.4f}\t'
        'MAE {mae_errors.avg:.3f}\n'.format(
        label, loss=losses, mae_errors=mae_errors))

    if test:  
        with open('test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets, test_preds):
                writer.writerow((cif_id, target, pred))
        pass
    else:
        return mae_errors.avg
                                                

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    '''
    Saves a checkpoint and overwrites the best model when is_best = True
    '''
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


if __name__ == '__main__':
    main()
