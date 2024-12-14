import torch
import torch.nn as nn 
class KernelMetricNetwork(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(KernelMetricNetwork, self).__init__()
        print('Using', num_classes, 'classes predictions')
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.dropout(self.batch_norm1(self.relu(self.fc1(x))))
        x = self.dropout(self.batch_norm2(self.relu(self.fc2(x))))
        x = self.fc3(x)
        return x

    
    def get_embeddings(self,x):
        
        x = self.batch_norm1(self.relu(self.fc1(x)))
        x = self.batch_norm2(self.relu(self.fc2(x)))
        
        return(x)
