import torch
import argparse
from datetime import datetime

from load_data import load_dataset
from models.model import TextCNN

def train(args, train_iter):
    model = TextCNN(args.class_num, args.embedding_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0

    # load models, optimizer, start_iter
    if args.snapshot is not None and os.path.exists(args.snapshot):
        print('Pre-trained model detected.\nLoading model...')
        checkpoint = torch.load(args.snapshot_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])
        start_epoch = checkpoint['epoch']

    # get device
    device = torch.device(args.device)
    model = model.to(device)
    criterion = criterion.to(device)

    print('====   Training..   ====')
    for epoch in range(start_epoch, start_epoch+args.epoch_num):
        print('Epoch: %d\t\t' % (epoch, ), end='')
        loss_sum = 0
        start_time = datetime.now()
        for iter_num, batch in enumerate(train_iter):
            optimizer.zero_grad()
            output = model(batch.title_words)
            output = output.max(1)[1]
            loss = criterion(output, batch.cate1_id)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        print('Loss = {}\t\ttime: {}\t\t'.format(loss_sum, datetime.now()-start_time), end='')
        if args.snapshot_path is None:
            snapshot_path = 'snapshot/model_{}.pth'.format(epoch)
        checkpoint = {
            'model': model.state_dict(),
            'optim': optimizer.state_dict(),
            'epoch': start_epoch+epoch_num
        }
        torch.save(checkpoint, model_path)
        print('Model saved in {}'.format(snapshot_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='../../data', help='Path to dataset (default="data/")')
    parser.add_argument('--device', default='cpu', help='Device to use (default="cpu")')
    parser.add_argument('--snapshot', default=None, help='Path to save model to save (default="checkpoints/crnn.pth")')
    parser.add_argument('--snapshot_path', default=None, help='Path to save model (default="snapshot/model_{epoch}.pth")')
    parser.add_argument('--batch_size', type=int, default=128, help='Input batch size (default=128)')
    parser.add_argument('--epoch_num', type=int, default=50, help='Number of epochs to train for (default=50)')
    parser.add_argument('--check_epoch', type=int, default=10, help='Epoch to save and test (default=10)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for Optimizer (default=0.001)')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for Optimizer (default=0)')
    args = parser.parse_args()
    print(args)

    train_iter = load_dataset(args)
    train(args, train_iter)

