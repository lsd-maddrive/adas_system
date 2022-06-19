import torch
import os


def save_checkpoint(model, scheduler, optimizer, epoch, filename):
    torch.save({
        'epoch': epoch if not epoch is None else None,
        'model': model.state_dict() if model else None,
        'optimizer': optimizer.state_dict() if optimizer else None,
        'scheduler': scheduler.state_dict() if scheduler else None
    }, filename)


def load_checkpoint(model, scheduler=None, optimizer=None, filename: str = 'model'):
    print(f'Loading checkpoint from {os.path.abspath(filename)}')
    checkpoint = torch.load(filename, map_location='cpu')
    if checkpoint['model']:
        model.load_state_dict(checkpoint['model'])
        model.eval()
    else:
        print('[!] Model was not loaded!')

    if not optimizer is None and checkpoint['optimizer']:
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print('[!] Optimizer was not loaded!')

    if not optimizer is None and checkpoint['scheduler']:
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        print('[!] Scheduler was not loaded!')

    if 'epoch' in checkpoint and not checkpoint['epoch'] is None:
        epoch = checkpoint['epoch']
    else:
        print('[!] No info about epochs!')
        epoch = 0

    return model, optimizer, scheduler, epoch
