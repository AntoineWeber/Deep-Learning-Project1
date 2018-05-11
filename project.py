import torch
import math

from torch import optim
from torch import Tensor

from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

import numpy as np
import matplotlib.pyplot as plt

import dlc_bci as bci


# Load the dataset
train_input, train_target = bci.load(root =  './data_bci', train=True, one_khz=True)
test_input, test_target = bci.load(root =  './data_bci', train=False, one_khz=True)

# Compute mean and std along the 28 channels and expand it to match the dimension of the input data
train_moy = train_input.mean(dim=2).mean(dim=0).expand(train_input.size(2),train_input.size(0),train_input.size(1)).transpose(0,1).transpose(1,2)
train_std = train_input.std(dim=2).mean(dim=0).expand(train_input.size(2),train_input.size(0),train_input.size(1)).transpose(0,1).transpose(1,2)

# Center and reduce the train input data
train_input = train_input.sub_(train_moy).div(train_std)

# Center and reduce the test input data using the test mean and std
test_moy = test_input.mean(dim=2).mean(dim=0).expand(test_input.size(2),test_input.size(0),test_input.size(1)).transpose(0,1).transpose(1,2)
test_std = test_input.std(dim=2).mean(dim=0).expand(test_input.size(2),test_input.size(0),test_input.size(1)).transpose(0,1).transpose(1,2)
test_input = test_input.sub_(test_moy).div(test_std)

# Transform the tensors into Variables
train_input, test_input, train_target, test_target = Variable(train_input), Variable(test_input), Variable(train_target), Variable(test_target)


nb_train_signals = train_input.shape[0]
nb_channels = train_input.shape[1]
print('nb_train_signals = ', nb_train_signals, ', nb_channels = ', nb_channels)

# Create a module to flatten the signals
class Flatten(nn.Module):
    def forward(self,input):
        return input.view(input.size(0), -1)


# Definition of our architecture
def create_model_1d():
    return nn.Sequential(
        nn.Conv1d(in_channels = nb_channels, out_channels = 28, kernel_size = (5),stride=5),
        nn.SELU(),
        nn.BatchNorm2d(28),
        nn.Conv1d(in_channels = 28, out_channels = 20, kernel_size = (5),stride=5),
        nn.SELU(),
        nn.BatchNorm2d(20),
        nn.Dropout(0.3),
        nn.Conv1d(in_channels = 20, out_channels = 5, kernel_size = (5),stride=5),
        nn.SELU(),
    
        Flatten(),
        nn.BatchNorm1d(20),
        nn.Dropout(0.5),
        nn.Linear(20,2), 
        nn.Softmax(dim=1))


# Function to train the model
def train_model(model, train_data, train_labels, test_data, test_labels, mini_batch_size, verbose=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    nb_epochs = 100
    
    trainloss = []
    err_train = []
    err_test = []

    for e in range(0, nb_epochs):
        
        # adjust the LR after a given number of epochs
        if e == 40:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
            
        if e == 70:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
            
            
        
        for b in range(math.ceil(train_input.size(0)/mini_batch_size)):
            # Implementation using a mini_batch_size
            # if there is at least the number mini_batch_isze of left samples
            if ((b+1)*mini_batch_size <= train_input.size(0)):
                output = model(train_input.narrow(0, b*mini_batch_size, mini_batch_size))
                loss = criterion(output, train_target.narrow(0, b*mini_batch_size, mini_batch_size))
            # if not, take only what is left
            else:
                output = model(train_input.narrow(0, b*mini_batch_size, (train_input.size(0)%mini_batch_size)))
                loss = criterion(output, train_target.narrow(0, b*mini_batch_size, (train_input.size(0)%mini_batch_size)))
            model.zero_grad()
            loss.backward()
            optimizer.step()

        # if verbose, print the evolution of the network
        if verbose:
            model.eval()
            output = model(train_input)
            model.train()
            trainloss.append(float(criterion(output, train_target)))
            err_train.append(compute_nb_errors(model, train_data, train_labels, mini_batch_size))
            err_test.append(compute_nb_errors(model, test_data, test_labels, mini_batch_size))
            print('Epoch number {}/{} finished. Train accuracy : {:0.2f}%, Test accuracy : {:0.2f}% with a train loss = {:0.5f}'.format(e+1, nb_epochs,
                                                                                                            100-(100*err_train[e]/train_data.size(0)),
                                                                                                            100-(100*err_test[e]/test_data.size(0)),
                                                                                                            trainloss[e]))
    if verbose:
        acc_train = 100-(100*np.divide(err_train,train_data.size(0)))
        acc_test = 100-(100*np.divide(err_test,test_data.size(0)))
        plt.figure(0)
        plot1 = plt.plot(acc_train, label='Train accuracy')
        plot2 = plt.plot(acc_test, label='Test accuracy')
        plt.xlabel('epochs')
        plt.ylabel('Accuracy [%]')
        plt.legend()
        plt.show()
        plt.figure(1)
        plot3 = plt.plot(trainloss, label='Train loss')
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        


# Function to compute the number of errors
def compute_nb_errors(model, data_input, data_target, mini_batch_size):
    nb_data_errors = 0

    for b in range(math.ceil(data_input.size(0)/mini_batch_size)):
        # same condition as in train_model regarding the number of left samples
        if ((b+1)*mini_batch_size <= data_input.size(0)):
            output = model(data_input.narrow(0, b*mini_batch_size, mini_batch_size))
            # the assigned label being the maximum value which has been outputed
            _, predicted_classes = torch.max(output.data, 1)
            for k in range(0,mini_batch_size):
                # check if the assigned label matches the ground truth
                if data_target.data[b*mini_batch_size + k] != predicted_classes[k]:
                    nb_data_errors = nb_data_errors + 1
        else:
            output = model(data_input.narrow(0, b*mini_batch_size, (data_input.size(0)%mini_batch_size)))
            _, predicted_classes = torch.max(output.data, 1)
            for k in range(0,data_input.size(0)%mini_batch_size):
                if data_target.data[b*mini_batch_size + k] != predicted_classes[k]:
                    nb_data_errors = nb_data_errors + 1
        

    return nb_data_errors


model = create_model_1d()
# best results obtained with a batch_size of 40
mini_batch_size = 40
model.train()
print('Training the model...')
train_model(model, train_input, train_target, test_input, test_target,mini_batch_size, verbose=True)
print('Model trained. Computing error rate...')
model.eval()
nb_test_errors = compute_nb_errors(model, test_input, test_target, mini_batch_size)
nb_train_errors = compute_nb_errors(model, train_input, train_target, mini_batch_size)
print('test accuracy = {:0.2f}% and train accuracy = {:0.2f}%'.format((100-(100 * nb_test_errors) / test_input.size(0)),
                                                                      (100-(100 * nb_train_errors) / train_input.size(0))))
print('With a batch size of {}' .format(mini_batch_size))




############################################################################
######################## Baseline computation ##############################
############################################################################
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn
from scipy.signal import butter, lfilter

def preprocess_signal(train_input, test_input, train_target, test_target, fs):
    train_data = train_input.data.numpy()
    test_data = test_input.data.numpy()
    train_labels = train_target.data.numpy()
    test_labels = test_target.data.numpy()

    #bandpass filter between 40 and 400hz
    for i in range(len(train_data)):
        for j in range(nb_channels):
            train_data[i,j] = butter_bandpass_filter(train_data[i,j], 40,400,fs)
    for i in range(len(test_data)):
        for j in range(nb_channels):
            test_data[i,j] = butter_bandpass_filter(test_data[i,j], 40,400,fs)

    #rectification
    train_data = np.abs(train_data)
    test_data = np.abs(test_data)

    #lowpass filtering to maintain only the envelope of signal
    for i in range(len(train_data)):
        for j in range(nb_channels):
            train_data[i,j] = butter_lowpass_filter(train_data[i,j], 20, fs)
    for i in range(len(test_data)):
        for j in range(nb_channels):
            test_data[i,j] = butter_lowpass_filter(test_data[i,j], 20,fs)

    return train_data,test_data,train_labels,test_labels

def compute_features(train_data, test_data):
    
    MAV_train = np.mean(np.abs(train_data),axis=2) #mean absolute value
    MAV_test = np.mean(np.abs(test_data),axis=2)

    NSC_train = np.sum(np.diff(np.sign(np.diff(train_data,axis=2)),axis=2)!=0,axis=2) #number of slope changes
    NSC_test = np.sum(np.diff(np.sign(np.diff(test_data,axis=2)),axis=2)!=0,axis=2)

    WL_train = np.sum(np.abs(np.diff(train_data,axis=2)),axis=2) #waveform length
    WL_test = np.sum(np.abs(np.diff(test_data,axis=2)),axis=2)

    total_data_train = np.concatenate((MAV_train,NSC_train,WL_train),axis=1)
    total_data_test = np.concatenate((MAV_test,NSC_test,WL_test),axis=1)

    return total_data_train,total_data_test

def perform_LDA(total_data_test, total_data_train, train_labels, test_labels):
    LDA = LinearDiscriminantAnalysis()
    trainedLDA = LDA.fit(total_data_train, train_labels)
    score_train = trainedLDA.score(total_data_train, train_labels)
    score_test = trainedLDA.score(total_data_test, test_labels)

    scat1 = plt.bar([1],[score_train])
    scat2 = plt.bar([2],[score_test])
    plt.title('Train and Test accuracy for an LDA classifier')
    plt.legend((scat1,scat2),('Train','Test'))
    plt.ylabel('Accuracy [%]')

def perform_SVM(total_data_test, total_data_train, train_labels, test_labels):
    SVMclass = sklearn.svm.SVC(0.9)
    trainedSVM = SVMclass.fit(total_data_train, train_labels)
    score_train = trainedSVM.score(total_data_train, train_labels)
    score_test = trainedSVM.score(total_data_test, test_labels)

    scat1 = plt.bar([1],[score_train])
    scat2 = plt.bar([2],[score_test])
    plt.title('Train and Test accuracy for a C-SVM classifier')
    plt.legend((scat1,scat2),('Train','Test'))
    plt.ylabel('Accuracy [%]')

######### Functions taken from some Stack Overflow posts ##########
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
