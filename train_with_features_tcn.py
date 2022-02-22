import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from utils import data_loader, data_retrieval
from models.tcn import TCN
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_set_path', default='data/i3d_feature/train')
    parser.add_argument('--val_set_path', default="data/i3d_feature/val")
    parser.add_argument('--test_set_path', default="data/i3d_feature/test")
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--output_size', type=int, default=3)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--level_size', type=int, default=10)
    parser.add_argument('--k_size', type=int, default=2)
    parser.add_argument('--drop_out', type=int, default=0.1)
    parser.add_argument('--fc_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=200)
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
    k_size = args.k_size
    hidden_size = args.hidden_size
    level_size = args.level_size
    num_chans = [hidden_size] * (level_size - 1) + [input_size]
    dropout = args.drop_out
    fc_size = args.fc_size
    num_epochs = args.num_epochs
    learning_rate = 3 * 1e-5

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = TCN(input_size, output_size, num_chans, k_size, dropout, fc_size)
    print("1. Model TCN loaded")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_f = torch.nn.CrossEntropyLoss()

    print("2. Start training")
    model.to(device)
    model.train()
    n_total_steps = len(train_loader)
    PATH = "model_zoo/your_model_zoo/TCN.pkl"
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
            train_set.transpose_(2, 1)
            ground_truth = y.to(device)
            outputs = model(train_set)
            print(outputs.shape)
            loss = loss_f(outputs, ground_truth)

            optimizer.zero_grad()

            # backtrack
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                train_f1_score = f1_score(ground_truth.data.to('cpu'),
                                          outputs.data.to('cpu').max(1, keepdim=True)[1].squeeze(), average='macro')
                val_x = val_x.to(device)
                val_y = val_y.to(device)

                val_outputs = model(val_x)
                val_f1_score = f1_score(val_y.data.to('cpu'),
                                        val_outputs.data.to('cpu').max(1, keepdim=True)[1].squeeze(), average='macro')
                val_loss_ = loss_f(val_outputs, val_y)

                if max_val_f1_score is None:
                    max_val_f1_score = val_f1_score
                else:
                    if max_val_f1_score < val_f1_score:
                        max_val_f1_score = val_f1_score
                        torch.save(model.state_dict(), PATH)
                target_names = ['class 0', 'class 1', 'class 2']
                print(
                    f' epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_total_steps}, train loss {loss.item():.4f}, val loss {val_loss_.item()}, val f1-score {val_f1_score}')
                train_l += loss.item()
                val_l += val_loss_.item()
                f1_ += val_f1_score
                train_f1 += train_f1_score

            model.train()
        train_l = train_l / len(train_loader)
        val_l = val_l / len(train_loader)
        f1_ = f1_ / len(train_loader)
        train_f1 = train_f1 / len(train_loader)
        train_loss.append(train_l)
        val_loss.append(val_l)
        f1_score_.append(f1_)
        train_f1_score_.append(train_f1)

    print("2. Finished training")











