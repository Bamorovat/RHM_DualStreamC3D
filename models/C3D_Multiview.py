import torch
import torch.nn as nn

Debug = False


class DualStreamC3D(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, num_classes, pretrained=False):
        self.pretrained = pretrained
        super(DualStreamC3D, self).__init__()

        self.conv1_1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1_1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2_1 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2_1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a_1 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b_1 = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3_1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a_1 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b_1 = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4_1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a_1 = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b_1 = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5_1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.conv1_2 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1_2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2_2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2_2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a_2 = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b_2 = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3_2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a_2 = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b_2 = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4_2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a_2 = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b_2 = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5_2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(768*4*4, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.batch1_1 = nn.BatchNorm3d(32)
        self.batch2_1 = nn.BatchNorm3d(64)
        self.batch3a_1 = nn.BatchNorm3d(128)
        self.batch3b_1 = nn.BatchNorm3d(128)
        self.batch4a_1 = nn.BatchNorm3d(256)
        self.batch4b_1 = nn.BatchNorm3d(256)
        self.batch5a_1 = nn.BatchNorm3d(256)
        self.batch5b_1 = nn.BatchNorm3d(256)

        self.batch1_2 = nn.BatchNorm3d(32)
        self.batch2_2 = nn.BatchNorm3d(64)
        self.batch3a_2 = nn.BatchNorm3d(128)
        self.batch3b_2 = nn.BatchNorm3d(128)
        self.batch4a_2 = nn.BatchNorm3d(256)
        self.batch4b_2 = nn.BatchNorm3d(256)
        self.batch5a_2 = nn.BatchNorm3d(512)
        self.batch5b_2 = nn.BatchNorm3d(512)

        self.batch6 = nn.BatchNorm1d(4096)
        self.batch7 = nn.BatchNorm1d(4096)

        self.__init_weight()

        if self.pretrained:
            self.__load_pretrained_weights()

    def forward(self, x1, x2):

        # Layer 1
        x1 = self.relu(self.conv1_1(x1))
        x1 = self.batch1_1(x1)
        x1 = self.pool1_1(x1)

        x2 = self.relu(self.conv1_2(x2))
        x2 = self.batch1_2(x2)
        x2 = self.pool1_2(x2)
        x2 = torch.cat([x2, x1], dim=1)

        if Debug:
            print('layer 1_x1: ', x1.shape)
            print('layer 1_x2: ', x2.shape)

        # Layer 2
        x1 = self.relu(self.conv2_1(x1))
        x1 = self.batch2_1(x1)
        x1 = self.pool2_1(x1)

        x2 = self.relu(self.conv2_2(x2))
        x2 = self.batch2_2(x2)
        x2 = self.pool2_2(x2)
        x2 = torch.cat([x2, x1], dim=1)

        if Debug:
            print('layer 2_x1: ', x1.shape)
            print('layer 2_x2: ', x2.shape)

        # Layer 3
        x1 = self.relu(self.conv3a_1(x1))
        x1 = self.batch3a_1(x1)
        x1 = self.relu(self.conv3b_1(x1))
        x1 = self.batch3b_1(x1)
        x1 = self.pool3_1(x1)

        x2 = self.relu(self.conv3a_2(x2))
        x2 = self.batch3a_2(x2)
        x2 = self.relu(self.conv3b_2(x2))
        x2 = self.batch3b_2(x2)
        x2 = self.pool3_2(x2)
        x2 = torch.cat([x2, x1], dim=1)

        if Debug:
            print('layer 3_x1: ', x1.shape)
            print('layer 3_x2: ', x2.shape)

        # Layer 4
        x1 = self.relu(self.conv4a_1(x1))
        x1 = self.batch4a_1(x1)
        x1 = self.relu(self.conv4b_1(x1))
        x1 = self.batch4b_1(x1)
        x1 = self.pool4_1(x1)

        x2 = self.relu(self.conv4a_2(x2))
        x2 = self.batch4a_2(x2)
        x2 = self.relu(self.conv4b_2(x2))
        x2 = self.batch4b_2(x2)
        x2 = self.pool4_2(x2)
        x2 = torch.cat([x2, x1], dim=1)

        if Debug:
            print('layer 4_x1: ', x1.shape)
            print('layer 4_x2: ', x2.shape)

        # Layer 5
        x1 = self.relu(self.conv5a_1(x1))
        x1 = self.batch5a_1(x1)
        x1 = self.relu(self.conv5b_1(x1))
        x1 = self.batch5b_1(x1)
        x1 = self.pool5_1(x1)

        x2 = self.relu(self.conv5a_2(x2))
        x2 = self.batch5a_2(x2)
        x2 = self.relu(self.conv5b_2(x2))
        x2 = self.batch5b_2(x2)
        x2 = self.pool5_2(x2)
        x2 = torch.cat([x2, x1], dim=1)

        if Debug:
            print('layer 5_x1: ', x1.shape)
            print('layer 5_x2: ', x2.shape)

        # Layer fc6

        x1 = x1.view(-1, 256*4*4)
        x2 = x2.view(-1, 768*4*4)

        if Debug:
            print('layer FC6: ')
            print('layer 6_x1', x1.shape)
            print('layer 4_x2', x2.shape)

        x2 = self.relu(self.fc6(x2))
        # x2 = self.batch6(x2)
        x2 = self.dropout(x2)

        if Debug:
            print('layer FC7: ')
            print('layer 7_x1', x1.shape)
            print('layer 7_x2', x2.shape)

        x2 = self.relu(self.fc7(x2))
        # x2 = self.batch7(x2)
        x2 = self.dropout(x2)

        if Debug:
            print('layer FC8: ')
            print('layer 8_x1', x1.shape)
            print('layer 8_x2', x2.shape)

        logits = self.fc8(x2)

        return logits

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.conv1_1, model.conv1_2, model.conv2_1, model.conv2_2, model.conv3a_1, model.conv3a_2,
         model.conv3b_1, model.conv3b_2, model.conv4a_1, model.conv4a_2, model.conv4b_1, model.conv4b_2,
         model.conv5a_1, model.conv5a_2, model.conv5b_1, model.conv5b_2, model.fc6, model.fc7]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc8]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k

