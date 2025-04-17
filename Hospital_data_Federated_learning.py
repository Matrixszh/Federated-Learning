import urllib.request
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import matplotlib.pyplot as plt

# Download functions
def download_url(url, save_as):
    response = urllib.request.urlopen(url)
    data = response.read()
    file = open(save_as, 'wb')
    file.write(data)
    file.close()
    response.close()

def read_binary_file(file):
    f = open(file,'rb')
    block = f.read()
    return block.decode('utf-16')

def split_text_in_lines(text):
    return text.split('\r\n')

def split_by_tabs(line):
    return line.split('\t')

# Data URLs and download
names_link = 'https://archive.ics.uci.edu/ml/machine-learning-databases/acute/diagnosis.names'
data_link = 'https://archive.ics.uci.edu/ml/machine-learning-databases/acute/diagnosis.data'
diagnosis_names = 'diagnosis.names'
diagnosis_data = 'diagnosis.data'
download_url(names_link, diagnosis_names)
download_url(data_link, diagnosis_data)

# Parse functions
def parse_double(field):
    field = field.replace(',', '.')
    return float(field)

def parse_boolean(field):
    return 1. if field == 'yes' else 0.

def read_np_array(file = diagnosis_data):
    text = read_binary_file(file)
    lines = split_text_in_lines(text)
    rows = []
    for line in lines:
        if line == '': continue
        line = line.replace('\r\n', '')
        fields = split_by_tabs(line)
        row = []
        j = 0
        for field in fields:
            value = parse_double(field) if j == 0 else parse_boolean(field)
            row.append(value)
            j += 1
        rows.append(row)
    matrix = np.array(rows, dtype = np.float32)
    return matrix

def get_random_indexes(n):
    indexes = list(range(n))
    random_indexes = []
    for i in range(n):
        r = np.random.randint(len(indexes))
        random_indexes.append(indexes.pop(r))
    return random_indexes

def get_indexes_for_2_datasets(n, training = 80):
    indexes = get_random_indexes(n)
    train = int(training / 100. * n)
    return indexes[:train], indexes[train:]

# Read data
matrix = read_np_array()
n_samples, n_dimensions = matrix.shape

train_indexes, test_indexes = get_indexes_for_2_datasets(n_samples)
train_data = matrix[train_indexes]
test_data = matrix[test_indexes]

def print_dataset(name, data):
    print('Dataset {}. Shape: {}'.format(name, data.shape))
    print(data)

print_dataset('Train', train_data)
print_dataset('Test', test_data)

# Logistic Regression Model
input_size = 6
learning_rate = 0.01
num_iterations = 20000

class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def decide(y):
    return 1. if y >= 0.5 else 0.

decide_vectorized = np.vectorize(decide)

to_percent = lambda x: '{:.2f}%'.format(x)

def compute_accuracy(model, input, output):
    prediction = model(input).data.numpy()[:, 0]
    n_samples = prediction.shape[0] + 0.
    prediction = decide_vectorized(prediction)
    equal = prediction == output.data.numpy()
    return 100. * equal.sum() / n_samples

def get_input_and_output(data):
    input = Variable(torch.tensor(data[:, :6], dtype=torch.float32))
    output1 = Variable(torch.tensor(data[:, 6], dtype=torch.float32)).view(-1, 1)
    output2 = Variable(torch.tensor(data[:, 7], dtype=torch.float32)).view(-1, 1)
    return input, output1, output2

input, output1, output2 = get_input_and_output(train_data)
test_input, test_output1, test_output2 = get_input_and_output(test_data)

# Plot graphs
def plot_graphs(diagnosis_title, losses, accuracies):
    plt.plot(losses)
    plt.title(f"{diagnosis_title} - Training Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Training Loss")
    plt.show()
    plt.plot(accuracies)
    plt.title(f"{diagnosis_title} - Training Accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy (Percent %)")
    plt.show()

# Train the model
def train_model(diagnosis_title, input, output, test_input, test_output):
    model = LogisticRegression()
    criterion = torch.nn.BCELoss(size_average=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  
    losses = []
    accuracies = []
    n_samples, _ = input.shape
    for iteration in range(num_iterations):
            optimizer.zero_grad()
            prediction = model(input)
            loss = criterion(prediction, output)
            loss.backward()
            optimizer.step()
            if iteration % 500 == 0:
                train_acc = compute_accuracy(model, input, output)
                train_loss = loss.item()
                losses.append(train_loss)
                accuracies.append(train_acc)
                print('iteration={}, loss={:.4f}, train_acc={}'.format(iteration, train_loss, to_percent(train_acc)))
    plot_graphs(diagnosis_title, losses, accuracies)
    test_acc = compute_accuracy(model, test_input, test_output)
    print('\nTesting Accuracy = {}'.format(to_percent(test_acc)))
    return model

# Train for both diagnosis types
model = train_model('Inflammation of Urinary Bladder', input, output1, test_input, test_output1)
model = train_model('Nephritis of Renal Pelvis Origin', input, output2, test_input, test_output2)

# Federated Learning Setup
class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(6, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def copy(self):
        new_model = LogisticRegression()
        new_model.load_state_dict(self.state_dict())
        return new_model

def to_percent(x):
    return '{:.2f}%'.format(x)

# Federated Learning Configuration
n_hospitals = 4
iterations = 1000
worker_iterations = 5
learning_rate = 0.01

# Convert and split data
train_data = torch.tensor(train_data, dtype=torch.float32)
test_input = torch.tensor(test_input, dtype=torch.float32)
test_output1 = torch.tensor(test_output1, dtype=torch.float32)
test_output2 = torch.tensor(test_output2, dtype=torch.float32)

n_samples = train_data.shape[0]
samples_per_hospital = int((n_samples + 0.5) / n_hospitals)

hospital_features = []
hospital_targets1 = []
hospital_targets2 = []

for i in range(n_hospitals):
    data_slice = train_data[i * samples_per_hospital:(i + 1) * samples_per_hospital]
    features = data_slice[:, :6]
    targets1 = data_slice[:, 6][:, None]
    targets2 = data_slice[:, 7][:, None]
    hospital_features.append(features)
    hospital_targets1.append(targets1)
    hospital_targets2.append(targets2)

# Plot Federated Learning Graphs
def plot_federated_graphs(title, losses, accuracies):
    for i in range(n_hospitals):
        plt.plot(losses[i], label=f'Hospital {i}')
    plt.legend(loc='upper right')
    plt.title(f"{title} - Training Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    for i in range(n_hospitals):
        plt.plot(accuracies[i], label=f'Hospital {i}')
    plt.legend(loc='lower right')
    plt.title(f"{title} - Training Accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy (%)")
    plt.show()

def compute_federated_accuracy(model, x, y):
    with torch.no_grad():
        preds = model(x)
        pred_labels = (preds >= 0.5).float()
        acc = (pred_labels == y).float().mean()
    return 100 * acc.item()

def federated_learning(title, features_list, targets_list, test_x, test_y):
    model = LogisticRegression()
    criterion = nn.BCELoss()
    losses = [[] for _ in range(n_hospitals)]
    accuracies = [[] for _ in range(n_hospitals)]

    for iteration in range(iterations):
        local_models = [model.copy() for _ in range(n_hospitals)]
        optimizers = [optim.SGD(m.parameters(), lr=learning_rate) for m in local_models]

        for _ in range(worker_iterations):
            last_losses = []
            for i in range(n_hospitals):
                optimizers[i].zero_grad()
                output = local_models[i](features_list[i])
                loss = criterion(output, targets_list[i])
                loss.backward()
                optimizers[i].step()
                last_losses.append(loss.item())

        for i in range(n_hospitals):
            losses[i].append(last_losses[i])
            acc = compute_federated_accuracy(local_models[i], features_list[i], targets_list[i])
            accuracies[i].append(acc)

        # Federated averaging
        with torch.no_grad():
            avg_weight = sum([m.linear.weight.data for m in local_models]) / n_hospitals
            avg_bias = sum([m.linear.bias.data for m in local_models]) / n_hospitals
            model.linear.weight.data.copy_(avg_weight)
            model.linear.bias.data.copy_(avg_bias)

        if iteration % 100 == 0:
            print(f"Iteration {iteration}:")
            for i in range(n_hospitals):
                print(f"  Hospital {i} - Loss: {losses[i][-1]:.4f}, Accuracy: {accuracies[i][-1]:.2f}%")

    plot_federated_graphs(title, losses, accuracies)

# Run Federated Learning for both diagnosis titles
federated_learning('Inflammation of Urinary Bladder Federated', hospital_features, hospital_targets1, test_input, test_output1)
federated_learning('Nephritis of Renal Pelvis Origin Federated', hospital_features, hospital_targets2, test_input, test_output2)
