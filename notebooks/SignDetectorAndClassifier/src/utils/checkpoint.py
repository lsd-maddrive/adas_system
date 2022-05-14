import torch
import os
def saveCheckpoint(model, scheduler, optimizer, epoch, filename):
    torch.save({
        'epoch': epoch if epoch else None,
        'model': model.state_dict() if model else None,
        'optimizer': optimizer.state_dict() if optimizer else None,
        'scheduler': scheduler.state_dict() if scheduler else None
    }, filename)

def loadCheckpoint(model, scheduler, optimizer, filename):
    print(f'Loading checkpoint from {os.path.abspath(filename)}')
    checkpoint = torch.load(filename)
    if checkpoint['model']:
        model.load_state_dict(checkpoint['model'])
        model.eval()
    else:
        print('[!] Model was not loaded!')

    if checkpoint['optimizer']:
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print('[!] optimizer was not loaded!')

    if checkpoint['scheduler']:
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        print('[!] Scheduler was not loaded!')

    if checkpoint['epoch']:
        epoch = checkpoint['epoch']
    else:
        print('[!] No info about epochs!')
        epoch = 0

    return model, optimizer, scheduler, epoch
