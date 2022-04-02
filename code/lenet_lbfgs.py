import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torch.optim import LBFGS
import torch.optim as optim
import torch.nn.functional as functional
from data_loader_Q5 import get_dataset

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3,padding=1)
        self.conv2 = nn.Conv2d(3, 3, 3,padding=1)
        self.fc1   = nn.Linear(24, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 26)

    def forward(self, x):
        out = functional.relu(self.conv1(x))
        out = functional.max_pool2d(out, 2)
        out = functional.relu(self.conv2(out))
        out = functional.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = functional.relu(self.fc1(out))
        out = functional.relu(self.fc2(out))
        out = self.fc3(out)
        return out


#compute word accuracy 
def accuracy_word(y_hat,y,dataset,split):
  
  flag = True
  wrong, total = 0,0
  rang = len(y_hat)
  for idx in range(rang):
    if y_hat[idx] != y[idx]:
      flag= False
    if dataset.next_[split+idx] == -1:
      if flag == False:
        wrong+=1
        flag  = True
      total = total+1
  acc = 1 - wrong/total
  print("Accuracy (word):", acc)
  return acc

#testing model 
def test_model(test_loader,cuda,lenet,criterion,wtestingepoc,testingepoc,test,dataset,split,batch_size):
    
    test_y_hat_letters,test_y_letters=[],[]
    letter_accuracy= 0 

    for iterator, datapoint in enumerate(test_loader):
           
            r_loss = 0.0
            r_right = 0
            test_X = datapoint[0]
            test_Y = datapoint[1]

            if len(test_X)<batch_size:
                prev_flag = True
            else:
                prev_flag = False
            if prev_flag == True:
                test_X= test_X.view(len(test_X),1,16,8)
                test_X= test_X.repeat(1,3,1,1)
                test_Y = test_Y.view(len(test_Y),26)
            else:
                test_X= test_X.view(batch_size,1,16,8)
                test_X= test_X.repeat(1,3,1,1)
                test_Y = test_Y.view(batch_size,26)
            if cuda:
                test_X = test_X.cuda()
                test_Y = test_Y.cuda()
                
            labels=  torch.max(test_Y, 1)[1]

            with torch.no_grad():
                outputs = lenet(test_X)
    
            loss = criterion(outputs,labels)
  
            _, y_hats = torch.max(outputs, 1)

            test_y_letters.extend(labels.tolist())
            test_y_hat_letters.extend(y_hats.tolist())
            r_loss += loss.item() * test_X.size(0)
            r_right += torch.sum(y_hats == (labels.data))


            # epoch_loss = r_loss / len(test_Y)
            epoch_acc = r_right.double() / len(test_Y)
            letter_accuracy = letter_accuracy + len(test_Y)*epoch_acc
            #print("Letter accuracy =",epoch_acc)

    wtestingepoc.append(accuracy_word(test_y_hat_letters,test_y_letters,dataset,split))
    testingepoc.append(letter_accuracy/len(test))
    print("Testing accuracy = :",letter_accuracy/len(test) )


def main():

    # Model parameters
    batch_size = 1200
    epoch  = 100
    trainingepoc=[]
    testingepoc=[]
    wtrainingepoc=[]
    wtestingepoc=[]
    
    # initializing torch tensors
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = get_dataset()
    lenet =  LeNet().to(device)
    criterion = nn.CrossEntropyLoss()

    optim = LBFGS(lenet.parameters(), lr = 0.01)
    
    split = int(0.5 * len(dataset.data))
    train_data, test_data = dataset.data[:split], dataset.data[split:]
    train_target, test_target = dataset.target[:split], dataset.target[split:]
    train = data_utils.TensorDataset(torch.tensor(train_data).float(), torch.tensor(train_target).long())
    test = data_utils.TensorDataset(torch.tensor(test_data).float(), torch.tensor(test_target).long())
    
    # Define train and test loaders
    train_loader = data_utils.DataLoader(train,  
                                            batch_size=batch_size, 
                                            shuffle=True,
                                            sampler=None,  
                                            num_workers=5,  
                                            pin_memory=False, 
                                            )

    test_loader = data_utils.DataLoader(test,
                                            batch_size=batch_size,  
                                            shuffle=False,
                                            sampler=None,  
                                            num_workers=5,  
                                            pin_memory=False, 
                                            )

    for i in range(epoch):
        train_y_hat_letters = []
        train_y_letters=[]
        train_acc= 0
        if i%1==0:
            print("Epoch",i)
        for iterator, datapoint in enumerate(train_loader):
            r_loss = 0.0
            r_right = 0
            if iterator%25==0:
                print("Batch=", iterator)
            train_X = datapoint[0]
            train_Y = datapoint[1]
            
            if len(train_X)<batch_size:
                prev_flag = True
            else:
                prev_flag = False
            
            if prev_flag == True:
                train_X= train_X.view(len(train_X),1,16,8)
                train_X= train_X.repeat(1,3,1,1)
                train_Y = train_Y.view(len(train_Y),26)
            else:
                train_X= train_X.view(batch_size,1,16,8)
                train_X= train_X.repeat(1,3,1,1)
                train_Y = train_Y.view(batch_size,26)
            
            if cuda:
                train_X = train_X.cuda()
                train_Y = train_Y.cuda()
            
            labels=  torch.max(train_Y, 1)[1]


            def closure():
                outputs = lenet(train_X)
                loss = criterion(outputs, labels)

                # losses_train.append(loss.item())
        
                # outputs_train.append(outputs)

                optim.zero_grad()
                loss.backward()

                # torch.nn.utils.clip_grad_norm_(lenet.parameters(), max_norm=3.0, norm_type=2)
                torch.nn.utils.clip_grad_value_(lenet.parameters(), clip_value=3.0)
                

                return loss


            # optim.zero_grad()
            # outputs = lenet(train_X)
            # loss = criterion(outputs, labels)
            # loss.backward()

            losses_train = optim.step(closure).item()
            # print("optim.step(closure)",losses_train)


            # optim.step()
            # print("Batch=", iterator)

            
            with torch.no_grad():
                # print("getting training predictions")
                outputs_train = lenet(train_X)
                
                _, y_hats = torch.max(outputs_train, 1)
                # print("y_hats",y_hats)

   


            train_y_letters.extend(labels.tolist())
            train_y_hat_letters.extend(y_hats.tolist())
            r_loss += losses_train * train_X.size(0)
            r_right += torch.sum(y_hats == (labels).data)

            print("running_corrects",r_right)

            # epoch_loss = r_loss / len(train_Y)
            epoch_acc = r_right.double() / len(train_Y)
            train_acc = train_acc + len(train_X)*epoch_acc

            if iterator%25==0:
                print("Accuracy (letters):",epoch_acc)

        # append the model accuracy for training
        wtrainingepoc.append(accuracy_word(train_y_hat_letters,train_y_letters,dataset,split)) 
        trainingepoc.append(train_acc/len(train))
        print("Training acc = :",train_acc/len(train))

        # test model 
        test_model(test_loader,cuda,lenet,criterion,wtestingepoc,testingepoc,test,dataset,split,batch_size)

        f_wtrainingepoc = open("lbfgs_files/wordwise_training.txt", "a")
        f_wtrainingepoc.write(str(wtrainingepoc[i]) + "\n")
        f_wtrainingepoc.close()

        f_trainingepoc = open("lbfgs_files/letterwise_training.txt", "a")
        f_trainingepoc.write(str(trainingepoc[i]) + "\n")
        f_trainingepoc.close()
    
        f_wtestingepoc = open("lbfgs_files/wordwise_testing.txt", "a")
        f_wtestingepoc.write(str(wtestingepoc[i]) + "\n")
        f_wtestingepoc.close()

        f_testingepoc = open("lbfgs_files/letterwise_testing.txt", "a")
        f_testingepoc.write(str(testingepoc[i]) + "\n")
        f_testingepoc.close()

if __name__ == "__main__":
    main()