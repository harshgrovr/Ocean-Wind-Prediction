import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch
from config import config

# Dense Network with 3 Linear layer
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(11 * 17, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = F.relu(self.fc1(x))        
        x = F.relu(self.fc2(x))        
        x = self.fc3(x)
        return x


# Pre-trained CNN(Resnet50) with 2 Fully connected layers at the end
class CNN(nn.Module):
    def __init__(self, train_CNN=False, num_classes=1):
        super(CNN, self).__init__()
        self.train_CNN = train_CNN
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, 256)
        self.Linear1 = nn.Linear(256, 32)
        self.Linear2 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, images):
        features = self.resnet50(images)
        features  = self.relu(self.Linear1(features))
        features = self.Linear2(features)
        return features

# Normal CNN network with 3 Conv Layers with Batch norm, 2 pooling layers
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 1 input image channel, 8 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(8, 16,kernel_size=3,stride=1, padding=1)        
        self.conv3 = nn.Conv2d(16, 32,kernel_size=3,stride=1, padding=1)           
        # 2*2 Pooling layer     
        self.pool = nn.MaxPool2d(2, 2)        
        # batch norm layers
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.batchnorm3 = nn.BatchNorm2d(32) 
        
    def forward(self, x):
        x = self.pool(F.relu(self.batchnorm1(self.conv1(x))))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = self.pool(F.relu(self.batchnorm3(self.conv3(x))))
        return x

# Normal CNN network with Simple CNN() network and one fully connected layer at the end
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()        
        self.SimpleCNN = SimpleCNN()           
        self.dropout = nn.Dropout(0.5)  
        self.fc1 = nn.Linear(32 * 4 * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.SimpleCNN(x)      
        x = x.view(-1, 32 * 4 * 2) # Flatten layer                
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# CNN-LSTM network with Simple CNN() network, one LSTM layer and two fully connected layer at the end.
class CNNLSTM(nn.Module):
    def init_hidden(self, batch_size):
        self.batch_size = batch_size
               
        if torch.cuda.is_available():
            # Initialize hidden state
            hn = torch.zeros(self.layer_dim , self.batch_size, self.hidden_dim).cuda()
            # Initialize cell state
            cn = torch.zeros(self.layer_dim, self.batch_size, self.hidden_dim).cuda()
        else:
            # Initialize hidden state
            hn = torch.zeros(self.layer_dim, self.batch_size, self.hidden_dim)
            # Initialize cell state
            cn = torch.zeros(self.layer_dim, self.batch_size, self.hidden_dim)
       
        return hn, cn

    def __init__(self, num_classes=1):
        super(CNNLSTM, self).__init__()        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.layer_dim = config['layer_dim']
        self.hidden_dim = config['hidden_dim']
        # CNN Network
        self.SimpleCNN = SimpleCNN()
        # LSTM layer
        self.lstm = nn.LSTM(256, hidden_size = self.hidden_dim, num_layers= self.layer_dim, 
                                batch_first=True)
        self.dropout = nn.Dropout(0.4) 
        # Linear layers                        
        self.linear_1 = nn.Linear(config['hidden_dim'], 32)
        self.linear_2 = nn.Linear(32, 1)
        
        

    # Defining the forward pass NTCHW
    def forward(self, x):
        # Initialize cell state and hidden state        
        self.hn, self.cn = self.init_hidden(config['batch_size'])        
        # Get Image dimension(batch, time_sequence, channel, height, width)
        batch_size, timesteps, C, H, W = x.size()             
        # Change dimensions for giving it to CNN(Merge batch and time dimension to make it a 4 dimensional tensor)
        c_in = x.view(batch_size * timesteps, C, H, W)                      
        # Give it to CNN
        c_out = self.SimpleCNN(c_in)
        # Change dimensions for giving it to LSTM
        r_in = c_out.view(batch_size, timesteps, -1)                                                                         
        r_out,(_,_) = self.lstm(r_in, None) 
        # Give output of LSTM to Fullconnected layers              
        r_out2 =  self.linear_2(self.dropout(F.relu(self.linear_1(self.dropout(r_out[:, -1, :])))))
        return r_out2 
    
    
    
    
    
    