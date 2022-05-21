import torch
from tqdm.notebook import tqdm
import numpy as np

from pytorch_metric_learning import losses, miners

@torch.no_grad()
def simpleGetAllEmbeddings(model, dataset, batch_size, dsc=''):

    dataloader = getDataLoaderFromDataset(
        dataset,
        shuffle=True,
        drop_last=False
    )

    s, e = 0, 0
    pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        position=0,
        leave=False,
        desc='Getting all embeddings...' + dsc)
    info_arr = []

    add_info_len = None

    for idx, (data, labels, info) in pbar:
        data = data.to(device)

        q = model(data)

        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        if idx == 0:
            labels_ret = torch.zeros(
                len(dataloader.dataset),
                labels.size(1),
                device=device,
                dtype=labels.dtype,
            )
            all_q = torch.zeros(
                len(dataloader.dataset),
                q.size(1),
                device=device,
                dtype=q.dtype,
            )

        info = np.array(info)
        if add_info_len == None:
            add_info_len = info.shape[0]

        info_arr.extend(info.T.reshape((-1, add_info_len)))
        e = s + q.size(0)
        all_q[s:e] = q
        labels_ret[s:e] = labels
        s = e

    all_q = torch.nn.functional.normalize(all_q)
    return all_q, labels_ret, info_arr

### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
@torch.no_grad()
def test(train_set, test_set, model, accuracy_calculator, batch_size):
    model.eval()
    train_embeddings, train_labels, _ = simpleGetAllEmbeddings(model, train_set, batch_size, ' for train')
    test_embeddings, test_labels, _ = simpleGetAllEmbeddings(model, test_set, batch_size, ' for test')
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, train_embeddings, test_labels, train_labels, False
    )
    print(accuracies)
    # print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))
    return accuracies["precision_at_1"]


### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ###
def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()
    loss_sum = 0

    pbar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        position=0,
        leave=False,
        desc='WAITING...')

    USING_CentroidTripletLoss_FLAG = False
    USING_MultiSimilarityMiner_FLAG = False
    if isinstance(loss_func, losses.CentroidTripletLoss):
        USING_CentroidTripletLoss_FLAG = True
    if isinstance(mining_func, miners.MultiSimilarityMiner):
        USING_MultiSimilarityMiner_FLAG = True

    for batch_idx, (data, labels, _) in pbar:

        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)

        if USING_CentroidTripletLoss_FLAG:
            embeddings = torch.tensor(
                [c_f.angle_to_coord(a) for a in embeddings],
                requires_grad=True,
                dtype=dtype,
            ).to(
                device
            )
            print(embeddings.shape)
            print(labels.shape)
            loss = loss_func(embeddings, labels)
        else:
            indices_tuple = mining_func(embeddings, labels)
            loss = loss_func(embeddings, labels, indices_tuple)

        instant_loss = loss.item()
        loss_sum += instant_loss

        loss.backward()
        optimizer.step()

        try:
            pbar.set_description("TRAIN: INSTANT MEAN LOSS %f, MINED TRIPLET: %d" %
                             (round(instant_loss / len(labels), 3),
                             mining_func.num_triplets)
                            )
        except:
            pbar.set_description("TRAIN: INSTANT MEAN LOSS %f" %
                             (round(instant_loss / len(labels), 3))
                            )


    return loss_sum / (train_loader.batch_size * len(train_loader))
