import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from utils import data_loader, data_retrieval
from models.spatial_temporal_model import efficientnet_tcn
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_set_path', default='data/images/train')
    parser.add_argument('--val_set_path', default="data/images/val")
    parser.add_argument('--test_set_path', default="data/images/test")
    parser.add_argument('--output_size', type=int, default=3)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--level_size', type=int, default=10)
    parser.add_argument('--k_size', type=int, default=2)
    parser.add_argument('--drop_out', type=int, default=0.1)
    parser.add_argument('--fc_size', type=int, default=128)
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
        print("Please check your image file path")

    print(f'0. received train image size: {train_images.shape}')
    print(f'0. received val image size: {val_images.shape}')
    print(f'0. received test image size: {test_images.shape}')

    val_x = torch.from_numpy(val_images).float()
    val_y = torch.tensor(val_labels, dtype=torch.float).long()

    test_x = torch.from_numpy(test_images).float()
    test_y = torch.tensor(test_labels, dtype=torch.float).long()

    train_dataset = data_loader.NumpyDataset(train_images, train_labels)
    batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = data_loader.NumpyDataset(val_images, val_labels)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    print("0. Finished dataset preparation")

    fc_size = args.fc_size
    output_size = args.output_size
    hidden_size = args.hidden_size
    dropout = args.drop_out
    k_size = args.k_size
    level_size = args.level_size

    num_epochs = args.num_epochs
    learning_rate = 3 * 1e-5

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = efficientnet_tcn(output_size, hidden_size, fc_size,dropout, k_size, level_size)
    print("1. Model TCN loaded")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_f = torch.nn.CrossEntropyLoss()

    model.to(device)
    n_total_steps = len(train_loader)
    max_val_f1_score = None
    PATH = "model_zoo/your_model_zoo/train_with_images.pkl"
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
            model.train()
            train_set = x.to(device)
            ground_truth = y.to(device)
            outputs = model(train_set)
            print(outputs.shape)
            loss = loss_f(outputs, ground_truth)

            optimizer.zero_grad()

            # backtrack
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                if len(ground_truth) > 1:
                    train_f1_score = f1_score(ground_truth.data.to('cpu'),
                                              outputs.data.to('cpu').max(1, keepdim=True)[1].squeeze(),
                                              average='weighted')
                else:
                    train_f1_score = 1
                counter_val_loss = 0
                predict_labels = []
                for j, (val_input, val_class) in enumerate(val_loader):
                    val_x = val_input.to(device)
                    val_y = val_class.to(device)

                    val_outputs = model(val_x)
                    val_loss_ = loss_f(val_outputs, val_y)
                    counter_val_loss += val_loss_.item()
                    val_outputs = val_outputs.data.to('cpu').max(1, keepdim=True)[1].squeeze()
                    val_outputs = val_outputs.tolist()
                    if isinstance(val_outputs, int):
                        predict_labels.append(val_outputs)
                    else:
                        predict_labels.extend(val_outputs)

                counter_val_loss = counter_val_loss / len(val_loader)
                counter_val_f1_score = f1_score(val_labels, predict_labels, average='weighted')

                if max_val_f1_score is None:
                    max_val_f1_score = counter_val_loss
                else:
                    if max_val_f1_score < counter_val_f1_score:
                        max_val_f1_score = counter_val_f1_score
                        torch.save(model.state_dict(), PATH)
                #
                print(
                    f' epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_total_steps}, train loss {loss.item():.4f}, val loss {val_loss_.item()}, val f1-score {counter_val_f1_score} ')
                train_l += loss.item()
                val_l += counter_val_loss
                f1_ += counter_val_f1_score
                train_f1 += train_f1_score

        train_l = train_l / len(train_loader)
        val_l = val_l / len(train_loader)
        f1_ = f1_ / len(train_loader)
        train_f1 = train_f1 / len(train_loader)
        train_loss.append(train_l)
        val_loss.append(val_l)
        f1_score_.append(f1_)
        train_f1_score_.append(train_f1)

        print("2. Finished training")

