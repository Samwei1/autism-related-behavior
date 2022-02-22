import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from utils import data_loader, data_retrieval
from models.ms_tcn.ms_tcn import MultiStageTCN
from models.ms_tcn.ms_tcn_loss import AverageMeter, ActionSegmentationLoss
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_set_path', default='data/i3d_feature/train')
    parser.add_argument('--val_set_path', default="data/i3d_feature/val")
    parser.add_argument('--test_set_path', default="data/i3d_feature/test")
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--output_size', type=int, default=3)
    parser.add_argument('--num_stages', type=list, default=['dilated', 'dilated', 'dilated', 'dilated'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=100)
    args = parser.parse_args()
    # -------------------- read features -------------------
    print("0. Begin to read features")
    PATH_train = args.train_set_path
    PATH_val = args.val_set_path
    PATH_test = args.test_set_path
    try:
        train_images, train_labels = data_retrieval.get_feature(PATH_train)
        val_images, val_labels = data_retrieval.get_feature(PATH_val)
        test_images, test_labels = data_retrieval.get_feature(PATH_test)
    except:
        print("Please check your feature file path")

    print(f'0. received train image size: {train_images.shape}')
    print(f'0. received val image size: {val_images.shape}')
    print(f'0. received test image size: {test_images.shape}')

    val_x = torch.from_numpy(val_images).float()
    val_y = torch.tensor(val_labels, dtype=torch.float).long()
    val_x.transpose_(2, 1)

    test_x = torch.from_numpy(test_images).float()
    test_y = torch.tensor(test_labels, dtype=torch.float).long()
    test_x.transpose_(2, 1)

    train_dataset = data_loader.NumpyDataset(train_images, train_labels)
    batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    print("0. Finished dataset preparation")

    input_size = args.input_size
    output_size = args.output_size
    num_stages = args.num_stages
    num_epochs = args.num_epochs
    learning_rate = 3 * 1e-5

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = MultiStageTCN(input_size, output_size, num_stages)
    print("1. Model TCN loaded")
    losses = AverageMeter('Loss', ':.4e')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = ActionSegmentationLoss(
        ce=True, tmse=True, weight=None,
        ignore_index=255, tmse_weight=0.15
    )

    print("2. Start training")
    model.to(device)
    model.train()
    n_total_steps = len(train_loader)
    max_val_f1_score = None
    PATH = "model_zoo/your_model_zoo/ms_tcn.pkl"
    max_val_f1_score = None

    train_loss = []
    val_loss = []
    f1_score_ = []
    train_f1_score_ = []
    for epoch in range(num_epochs):
        train_l = 0
        val_l = 0
        f1_ = 0
        train_f1 = 0
        for i, (x, y) in enumerate(train_loader):
            # forward
            train_set = x.to(device)
            ground_truth = np.zeros((len(x), 12))
            for k in range(len(x)):
                ground_truth[k, :] = np.array([y[k] for _ in range(12)])
            ground_truth = torch.tensor(ground_truth).to(device)
            train_set.transpose_(2, 1)
            output = model(train_set)

            if isinstance(output, list):
                loss = 0.0
                for out in output:
                    ground_truth = ground_truth.type(torch.LongTensor).to(device)
                    loss += criterion(out, ground_truth, train_set)
            else:
                loss = criterion(output, ground_truth.type(torch.LongTensor), train_set)

            optimizer.zero_grad()

            # backtrack
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.eval()
                val_x_ = val_x.to(device)
                val_output = model(val_x_)
                output_ = val_output.max(1)[1].squeeze(0).cpu().numpy()
                pre = []
                for i in range(len(output_)):
                    kk = output_[i]
                    counts = np.bincount(kk)
                    pre.append(np.argmax(counts))
                val_f1_score = f1_score(val_y, pre, average='weighted')
                if max_val_f1_score is None:
                    max_val_f1_score = val_f1_score
                else:
                    if max_val_f1_score < val_f1_score:
                        max_val_f1_score = val_f1_score
                        torch.save(model.state_dict(), PATH)
                model.train()

            print(
                f'epoch: {epoch} batch: {i} / {len(train_loader)} loss: {loss.item()} val weighted f1-score: {val_f1_score}')

        print("2. Finished training")


