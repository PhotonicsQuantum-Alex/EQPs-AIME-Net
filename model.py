import torch.nn as nn
import torch.nn.init as init


class MyNet_MLP_Improved_GELU_2qubit(nn.Module):
    def __init__(self, nProj):
        super(MyNet_MLP_Improved_GELU_2qubit, self).__init__()
        self.nProj = nProj
        input_dim = nProj * 17
        self.gelu = nn.GELU()

        hidden1 = 512
        hidden2 = 256
        hidden3 = 128

        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, 12)

        self.res_proj1 = nn.Linear(input_dim, hidden1) if input_dim != hidden1 else nn.Identity()
        self.res_proj2 = nn.Linear(hidden1, hidden2) if hidden1 != hidden2 else nn.Identity()
        self.res_proj3 = nn.Linear(hidden2, hidden3) if hidden2 != hidden3 else nn.Identity()

        self.layer_norm_in = nn.LayerNorm(input_dim)
        self.layer_norm1 = nn.LayerNorm(hidden1)
        self.layer_norm2 = nn.LayerNorm(hidden2)
        self.layer_norm3 = nn.LayerNorm(hidden3)

        self.dropout = nn.Dropout(0.2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):
        x = x.view(x.size(0), self.nProj, 17).view(x.size(0), -1)
        x = self.layer_norm_in(x)

        res = self.res_proj1(x)
        x = self.gelu(self.fc1(x))
        x = self.layer_norm1(x + res)
        x = self.dropout(x)

        res = self.res_proj2(x)
        x = self.gelu(self.fc2(x))
        x = self.layer_norm2(x + res)
        x = self.dropout(x)

        res = self.res_proj3(x)
        x = self.gelu(self.fc3(x))
        x = self.layer_norm3(x + res)
        x = self.dropout(x)

        x = self.fc4(x)
        return x

class MyNet_MLP_Improved_GELU_3qubit(nn.Module):
    def __init__(self, nProj):
        super(MyNet_MLP_Improved_GELU_3qubit, self).__init__()
        self.nProj = nProj
        input_dim = nProj * 65
        self.gelu = nn.GELU()

        hidden1 = min(4096, int(input_dim * 0.05))
        hidden2 = hidden1 // 2
        hidden3 = hidden2 // 2

        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, 25)

        self.res_proj1 = nn.Linear(input_dim, hidden1) if input_dim != hidden1 else nn.Identity()
        self.res_proj2 = nn.Linear(hidden1, hidden2) if hidden1 != hidden2 else nn.Identity()
        self.res_proj3 = nn.Linear(hidden2, hidden3) if hidden2 != hidden3 else nn.Identity()

        self.layer_norm_in = nn.LayerNorm(input_dim)
        self.layer_norm1 = nn.LayerNorm(hidden1)
        self.layer_norm2 = nn.LayerNorm(hidden2)
        self.layer_norm3 = nn.LayerNorm(hidden3)

        self.dropout = nn.Dropout(0.2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):
        x = x.view(x.size(0), self.nProj, 65).view(x.size(0), -1)
        x = self.layer_norm_in(x)

        res = self.res_proj1(x)
        x = self.gelu(self.fc1(x))
        x = self.layer_norm1(x + res)
        x = self.dropout(x)

        res = self.res_proj2(x)
        x = self.gelu(self.fc2(x))
        x = self.layer_norm2(x + res)
        x = self.dropout(x)

        res = self.res_proj3(x)
        x = self.gelu(self.fc3(x))
        x = self.layer_norm3(x + res)
        x = self.dropout(x)

        x = self.fc4(x)
        return x