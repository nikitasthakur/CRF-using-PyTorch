import os
import math
import gzip
import csv
import time
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split

from tqdm import tqdm

# import matplotlib.pyplot as plt
import numpy as np
from crf import CRF

# import Data Loader
from data_loader import get_dataset

if __name__ == '__main__':
    # hyperparameters, dimensions and model parameters
    dim = 128
    epochs = 1
    labels = 26
    max_iter = 500
    embed_dim = 128
    batch_size = 64
    conv_shapes = [[1, 64, 128]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model and optimizer
    model = CRF(dim, embed_dim, conv_shapes, labels, batch_size).to(device)
    opt = optim.LBFGS(model.parameters(), lr=0.01)

    dataset = get_dataset()

    print(dataset.target.shape, dataset.data.shape)
    # X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, stratify=dataset.target)
    split = int(0.7 * len(dataset.data))
    X_train, X_test = dataset.data[:split], dataset.data[split:]
    y_train, y_test = dataset.target[:split], dataset.target[split:]
    # train_data = train_data.to(device)
    # test_data = test_data.to(device)
    # train_target = train_target.to(device)
    # test_target = test_target.to(device)

    train = data_utils.TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
    test = data_utils.TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float())
    # train = train.to(device)
    # test = test.to(device)
    # print(len(train[0][1][0]))
    train_letter, test_letter, train_word, test_word = [], [], [], []

    # Clear all log files
    dir_name = "Q4"
    files = os.listdir(dir_name)

    for file in files:
        if file.endswith(".txt"):
            with open(os.path.join(dir_name, file), "r+") as f:
                f.truncate(0)
                f.close()

    for i in range(epochs):
        step = 1
        print("\nEpoch {}".format(i + 1))
        start_epoch = time.time()

        train_data = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True, sampler=None, num_workers=5,
                                           pin_memory=True)
        test_data = data_utils.DataLoader(test, batch_size=batch_size, shuffle=True, sampler=None, num_workers=5,
                                          pin_memory=True)
        train_mean_word_accuracy, test_mean_word_accuracy, train_mean_letter_accuracy, test_mean_letter_accuracy = 0, 0, 0, 0

        for batch, sample in tqdm(enumerate(train_data)):
            print("\nEpoch-{} Mini-Batch-{}".format(i + 1, batch))
            start_t = time.time()
            train_X = sample[0].to(device)
            train_Y = sample[1].to(device)


            def compute_loss():
                opt.zero_grad()
                _loss = model.loss(train_X, train_Y)
                _loss.backward()
                return _loss


            start_step = time.time()
            opt.step(compute_loss)
            print("Epoch-{} Batch-{} Step-{} TIME ELAPSED = {}".format(i + 1, batch, step, time.time() - start_step))
            for name, values in model.named_parameters():
                if values.requires_grad:
                    print("Parameters", name, values.data)

            random_index = np.random.choice(X_test.shape[0], batch_size, replace=False)
            test_X = X_test[random_index, :]
            test_Y = y_test[random_index, :]
            test_X = torch.from_numpy(test_X).float().to(device)
            test_Y = torch.from_numpy(test_Y).long().to(device)
            total_train_words = len(train_Y)
            total_test_words = len(test_Y)

            total_train_letters = torch.sum(train_Y).item()
            total_test_letters = torch.sum(test_Y).item()

            print("Getting Accuracy")
            with torch.no_grad():
                print("Training predictions-->")
                train_predictions = model(train_X)
                print("Test predictions-->")
                test_predictions = model(test_X)

            word_acc_train = 0
            letter_acc_train = 0

            for y, y_preds in zip(train_Y, train_predictions):
                letters = int(torch.sum(y).item())
                if torch.all(torch.eq(y[:letters], y_preds[:letters])):
                    word_acc_train = word_acc_train + 1
                letter_acc_train = letter_acc_train + letters - (
                            ((~torch.eq(y[:letters], y_preds[:letters])).sum()) / 2).item()

            word_accuracy_test = 0
            letter_accuracy_test = 0

            for y, y_preds in zip(test_Y, test_predictions):
                letters = int(torch.sum(y).item())
                if torch.all(torch.eq(y[:letters], y_preds[:letters])):
                    word_accuracy_test = word_accuracy_test + 1
                letter_accuracy_test = letter_accuracy_test + letters - (
                            ((~torch.eq(y[:letters], y_preds[:letters])).sum()) / 2).item()

            letter_acc_train /= total_train_letters
            letter_accuracy_test /= total_test_letters
            word_acc_train /= total_train_words
            word_accuracy_test /= total_test_words

            ## collect accuracies for 100 steps
            train_letter.append(letter_acc_train)
            test_letter.append(letter_accuracy_test)
            train_word.append(word_acc_train)
            test_word.append(word_accuracy_test)

            f_trainingepoc = open("Q4/wordwise_training.txt", "a")
            f_trainingepoc.write(str(word_acc_train) + "\n")
            f_trainingepoc.close()

            f_trainingepoc = open("Q4/letterwise_training.txt", "a")
            f_trainingepoc.write(str(letter_acc_train) + "\n")
            f_trainingepoc.close()

            f_wtestingepoc = open("Q4/wordwise_testing.txt", "a")
            f_wtestingepoc.write(str(word_accuracy_test) + "\n")
            f_wtestingepoc.close()

            f_testingepoc = open("Q4/letterwise_testing.txt", "a")
            f_testingepoc.write(str(letter_accuracy_test) + "\n")
            f_testingepoc.close()

            print("\nTraining Accuracy ")
            print("\tWord Acc = ", train_word)
            print("\tLetter Acc = ", train_letter)
            print(" Test Accuracy : ")
            print("\tWord accuracy = ", test_word)
            print("\tLetter accuracy = ", test_letter)
            train_mean_word_accuracy = sum(train_word) / len(train_word)
            test_mean_word_accuracy = sum(test_word) / len(test_word)
            train_mean_letter_accuracy = sum(train_letter) / len(train_letter)
            test_mean_letter_accuracy = sum(test_letter) / len(test_letter)

            print(
                "\n Train mean word accuracy = {}\n Test mean word accuracy = {}\n Train mean letter accuracy = {}\n Test mean letter accuracy = {}\n".format(
                    train_mean_word_accuracy, test_mean_word_accuracy, train_mean_letter_accuracy,
                    test_mean_letter_accuracy))
            print("Epoch-{} Batch-{} Step-{} TIME TAKEN = {}".format(i, batch, step, time.time() - start_t))
            step += 1

            if step > max_iter: break

        print("Epoch completed Epoch-{} Batch-{} Step-{} TIME ELAPSED = {}".format(i + 1, batch, step - 1,
                                                                                   time.time() - start_epoch))