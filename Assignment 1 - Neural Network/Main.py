from Network import Network
from DataManager import split_train_test
import numpy


def construct_network(n_features, n_classes, n_hidden_layers, size_hidden_layers):
    numpy.random.seed(1)
    size_network = [n_features] + size_hidden_layers + [n_classes]
    return Network(size_network)


def main():

    dataset_name        = "banknote"
    raw_data_file       = "datasets/" + dataset_name + "/raw_data.csv"
    train_data_file     = "datasets/" + dataset_name + "/train.csv"
    test_data_file      = "datasets/" + dataset_name + "/test.csv"
    train_size          = 0.7
    test_size           = 0.3

    split_train_test(raw_data_file, train_data_file, test_data_file, train_size, test_size)

    network = construct_network(n_features=4, n_classes=2, n_hidden_layers=2, size_hidden_layers=[4, 4])

    network.train(filename=train_data_file, epoch=10000, activator="sigmoid", learning_rate=0.4)

    network.test(filename=test_data_file)

if __name__ == '__main__':
    main()
