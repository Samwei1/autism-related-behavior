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


