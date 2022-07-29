import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, training=True):
        super(UNet, self).__init__()
        self.training = training
        self.encoder1 = nn.Conv3d(in_channel, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2 = nn.Conv3d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3 = nn.Conv3d(64, 128, 3, stride=1, padding=1)
        self.encoder4 = nn.Conv3d(128, 256, 3, stride=1, padding=1)
        # self.encoder5=   nn.Conv3d(256, 512, 3, stride=1, padding=1)

        # self.decoder1 = nn.Conv3d(512, 256, 3, stride=1,padding=1)  # b, 16, 5, 5
        self.decoder2 = nn.Conv3d(256, 128, 3, stride=1, padding=1)  # b, 8, 15, 1
        self.decoder3 = nn.Conv3d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 = nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv3d(32, 2, 3, stride=1, padding=1)

        self.map4 = nn.Sequential(
            nn.Conv3d(2, out_channel, 1, 1),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear'),
            nn.Softmax(dim=1)
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(64, out_channel, 1, 1),
            nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear'),
            nn.Softmax(dim=1)
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(128, out_channel, 1, 1),
            nn.Upsample(scale_factor=(8, 16, 16), mode='trilinear'),
            nn.Softmax(dim=1)
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, out_channel, 1, 1),
            nn.Upsample(scale_factor=(16, 32, 32), mode='trilinear'),
            nn.Softmax(dim=1)
        )

    def forward(self, x):

        out = F.relu(F.max_pool3d(self.encoder1(x), 2, 2))
        t1 = out
        out = F.relu(F.max_pool3d(self.encoder2(out), 2, 2))
        t2 = out
        out = F.relu(F.max_pool3d(self.encoder3(out), 2, 2))
        t3 = out
        out = F.relu(F.max_pool3d(self.encoder4(out), 2, 2))
        # t4 = out
        # out = F.relu(F.max_pool3d(self.encoder5(out),2,2))

        # t2 = out
        # out = F.relu(F.interpolate(self.decoder1(out),scale_factor=(2,2,2),mode ='trilinear'))
        # print(out.shape,t4.shape)
        output1 = self.map1(out)
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2, 2), mode='trilinear'))
        out = torch.add(out, t3)
        output2 = self.map2(out)
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2, 2), mode='trilinear'))
        out = torch.add(out, t2)
        output3 = self.map3(out)
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2, 2), mode='trilinear'))
        out = torch.add(out, t1)

        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2, 2), mode='trilinear'))
        output4 = self.map4(out)
        # print(out.shape)
        # print(output1.shape,output2.shape,output3.shape,output4.shape)
        if self.training is True:
            return output1, output2, output3, output4
        else:
            return output4


'''
    输入的数据为 B * C * W * H * L

    编码器Encoder下采样 将数据卷积成 B * C * 1

    全连接层

    解码器Decoder上采样还原图像 恢复到 B * C * W * H * L

'''


class Model(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.button_channel = 128
        channel = [16, 32, 64, 128]
        self.encoder = Encoder(in_channel, channels=channel)
        self.decoder = Decoder(channels=channel, out_channel=out_channel)
        self.button = Button_model()
        # self.up = nn.Upsample(size=[224, 224, 16], mode='trilinear', align_corners=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        shape = x.shape
        # x = self.up(x)
        x, skip_connection = self.encoder(x)
        x = self.button(x)
        x = self.decoder(x, skip_connection, shape)
        # x = torch.round(x)
        x = self.softmax(x)
        return x


class Conv3d_block(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.Cov = nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=out_channel,
                      kernel_size=(3, 3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=out_channel, out_channels=out_channel,
                      kernel_size=(3, 3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.Cov(x)


# size=[1,224,224,16]
class Encoder(nn.Module):
    def __init__(self, in_channel, channels) -> None:
        super().__init__()
        self.Conv3D = nn.ModuleList()
        for channel in channels:
            self.Conv3D.append(Conv3d_block(in_channel=in_channel, out_channel=channel))
            in_channel = channel
        self.avg_pool3d = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

    def forward(self, x):
        skip_conection = []
        for down in self.Conv3D:
            x = down(x)
            # resnet
            x = self.avg_pool3d(x)
            skip_conection.append(x)
        return x, skip_conection


class Button_model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x


class CBAM(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x


class Trans_FCN_block(nn.Module):
    def __init__(self, inchannel) -> None:
        super().__init__()
        self.con3d = Conv3d_block(in_channel=inchannel * 2, out_channel=inchannel // 2)
        # self.Channel_down = nn.Conv3d(in_channels = inchannel * 2, out_channels = inchannel//2 ,kernel_size=(1,1,1))
        # nn.ConvTranspose3d(in_channels=inchannel, out_channels=inchannel)

    def forward(self, x, skip_connection):
        shapes = skip_connection.shape

        x = nn.Upsample(size=shapes[2:], mode='trilinear', align_corners=True)(x)
        x = torch.cat((x, skip_connection), dim=1)
        x = self.con3d(x)
        x = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)(x)
        return x


class Decoder(nn.Module):
    def __init__(self, channels, out_channel) -> None:
        super().__init__()
        self.up_sample = nn.ModuleList()
        self.channels = channels
        for channel in channels[::-1]:
            self.up_sample.append(Trans_FCN_block(channel))
        self.down_channel = nn.Conv3d(channels[0] // 2, out_channel, kernel_size=(1, 1, 1))

    def forward(self, x, skip_connections, shape):
        for skip_connection, up_sample in zip(skip_connections[::-1], self.up_sample):
            x = up_sample(x, skip_connection)
        x = nn.Upsample(size=shape[2:], mode='trilinear', align_corners=True)(x)
        x = self.down_channel(x)
        # x = self.softmax(x)
        return x


if __name__ == '__main__':
    x = torch.rand(size=[1, 1, 245, 245, 18])
    skip = torch.rand(size=[1, 256, 14, 14, 1])
    model = Model(1, 1)
    # button shape 256,13,13,1
    y = model(x)
    print(y.shape)
