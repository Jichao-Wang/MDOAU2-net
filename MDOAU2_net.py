import torch
from torch import nn
import stable_seed
import torch.nn.functional as F

stable_seed.setup_seed()


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class multi_scaled_dilation_conv_block(nn.Module):
    # 多尺度预处理kernel
    def __init__(self, ch_in, ch_out, kernel_size, dilation=1):
        super(multi_scaled_dilation_conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(int((kernel_size - 1) / 2 * dilation)),
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=1, dilation=dilation, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class bias_convolution(nn.Module):
    # 多方向的空洞卷积，提供每个像素不同方向的情况
    def __init__(self, ch_in, ch_out, kernel_size, dilation=1, direction=''):
        # default is normal convolution
        super(bias_convolution, self).__init__()
        self.direction = direction
        self.padding_size = int((kernel_size - 1) * dilation)
        # self.direction_padding = nn.ReflectionPad2d(self.padding_size)
        self.direction_padding_LU = nn.ReflectionPad2d((self.padding_size, 0, self.padding_size, 0))
        self.direction_padding_RU = nn.ReflectionPad2d((0, self.padding_size, self.padding_size, 0))
        self.direction_padding_LD = nn.ReflectionPad2d((self.padding_size, 0, 0, self.padding_size))
        self.direction_padding_RD = nn.ReflectionPad2d((0, self.padding_size, 0, self.padding_size))

        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=1, dilation=dilation, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # print(self.padding_size)
        # x = self.direction_padding(x)
        x_LU = self.direction_padding_LU(x)
        x_RU = self.direction_padding_RU(x)
        x_LD = self.direction_padding_LD(x)
        x_RD = self.direction_padding_RD(x)

        if self.direction == 'LU':
            # padding to left up
            return self.conv(x_LU)

        elif self.direction == 'LD':
            # padding to left down
            return self.conv(x_LD)

        elif self.direction == 'RU':
            # padding to right up
            return self.conv(x_RU)

        elif self.direction == 'RD':
            # padding to right down
            return self.conv(x_RD)

        else:
            # normal padding
            return self.conv(x)


class offset_convolution(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(offset_convolution, self).__init__()
        self.LU_bias_convolution = bias_convolution(ch_in=ch_in, ch_out=ch_out, kernel_size=7, dilation=1,
                                                    direction='LU')
        self.LD_bias_convolution = bias_convolution(ch_in=ch_in, ch_out=ch_out, kernel_size=7, dilation=1,
                                                    direction='LD')
        self.RU_bias_convolution = bias_convolution(ch_in=ch_in, ch_out=ch_out, kernel_size=7, dilation=1,
                                                    direction='RU')
        self.RD_bias_convolution = bias_convolution(ch_in=ch_in, ch_out=ch_out, kernel_size=7, dilation=1,
                                                    direction='RD')
        self.final_conv = nn.Conv2d(ch_out * 4, ch_out, kernel_size=3, stride=1, padding=1)
        self.BN = nn.BatchNorm2d(ch_out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        LU_BC = self.LU_bias_convolution(x)
        LD_BC = self.LD_bias_convolution(x)
        RU_BC = self.RU_bias_convolution(x)
        RD_BC = self.RD_bias_convolution(x)
        d = torch.cat((LU_BC, LD_BC, RU_BC, RD_BC), dim=1)
        d = self.final_conv(d)
        d = self.BN(d)
        d = self.activation(d)
        return d


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class MDOAU2_net_1(nn.Module):
    # delete attention block
    def __init__(self, img_ch=1, output_ch=1):
        super(MDOAU2_net_1, self).__init__()
        # offset_convolution()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.multi_scale_1 = multi_scaled_dilation_conv_block(img_ch, 16, kernel_size=3, dilation=1)
        self.multi_scale_2 = multi_scaled_dilation_conv_block(img_ch, 16, kernel_size=5, dilation=1)
        self.multi_scale_3 = multi_scaled_dilation_conv_block(img_ch, 16, kernel_size=7, dilation=2)
        self.multi_scale_4 = multi_scaled_dilation_conv_block(img_ch, 16, kernel_size=9, dilation=2)
        self.multi_scale_5 = multi_scaled_dilation_conv_block(img_ch, 16, kernel_size=11, dilation=3)

        self.Conv1 = conv_block(ch_in=16 * 5, ch_out=8)
        self.Conv2 = conv_block(ch_in=8, ch_out=16)
        self.Conv3 = conv_block(ch_in=16, ch_out=32)
        self.Conv4 = conv_block(ch_in=32, ch_out=64)
        self.Conv5 = conv_block(ch_in=64, ch_out=128)

        self.o1 = offset_convolution(ch_in=8, ch_out=8)
        self.o2 = offset_convolution(ch_in=16, ch_out=16)
        self.o3 = offset_convolution(ch_in=32, ch_out=32)
        self.o4 = offset_convolution(ch_in=64, ch_out=64)

        self.Up5 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv5 = conv_block(ch_in=128, ch_out=64)

        self.Up4 = up_conv(ch_in=64, ch_out=32)
        self.Up_conv4 = conv_block(ch_in=64, ch_out=32)

        self.Up3 = up_conv(ch_in=32, ch_out=16)
        self.Up_conv3 = conv_block(ch_in=32, ch_out=16)

        self.Up2 = up_conv(ch_in=16, ch_out=8)
        self.Up_conv2 = conv_block(ch_in=16, ch_out=8)

        self.Conv_1x1 = nn.Conv2d(8, output_ch, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.Conv_1x1_1 = nn.Conv2d(8, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x, train_flag=False):
        # multi_scale_generator
        x_pre_1 = self.multi_scale_1(x)
        x_pre_2 = self.multi_scale_2(x)
        x_pre_3 = self.multi_scale_3(x)
        x_pre_4 = self.multi_scale_4(x)
        x_pre_5 = self.multi_scale_5(x)
        muti_scale_x = torch.cat((x_pre_1, x_pre_2, x_pre_3, x_pre_4, x_pre_5), dim=1)

        # encoding path
        x1 = self.Conv1(muti_scale_x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # offset convolution
        o1 = self.o1(x1)
        o2 = self.o2(x2)
        o3 = self.o3(x3)
        o4 = self.o4(x4)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((o4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((o3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((o2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((o1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        if train_flag:
            return d1
        else:
            return self.sigmoid(d1)


class MDOAU2_net_2(nn.Module):
    # delete attention block
    # offset_convolution block take place conv_block
    def __init__(self, img_ch=1, output_ch=1):
        super(MDOAU2_net_2, self).__init__()
        # offset_convolution()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.multi_scale_1 = multi_scaled_dilation_conv_block(img_ch, 16, kernel_size=3, dilation=1)
        self.multi_scale_2 = multi_scaled_dilation_conv_block(img_ch, 16, kernel_size=5, dilation=1)
        self.multi_scale_3 = multi_scaled_dilation_conv_block(img_ch, 16, kernel_size=7, dilation=2)
        self.multi_scale_4 = multi_scaled_dilation_conv_block(img_ch, 16, kernel_size=9, dilation=2)
        self.multi_scale_5 = multi_scaled_dilation_conv_block(img_ch, 16, kernel_size=11, dilation=3)

        self.Conv1 = offset_convolution(ch_in=16 * 5, ch_out=8)
        self.Conv2 = offset_convolution(ch_in=8, ch_out=16)
        self.Conv3 = offset_convolution(ch_in=16, ch_out=32)
        self.Conv4 = offset_convolution(ch_in=32, ch_out=64)
        self.Conv5 = offset_convolution(ch_in=64, ch_out=128)

        self.o1 = offset_convolution(ch_in=8, ch_out=8)
        self.o2 = offset_convolution(ch_in=16, ch_out=16)
        self.o3 = offset_convolution(ch_in=32, ch_out=32)
        self.o4 = offset_convolution(ch_in=64, ch_out=64)

        self.Up5 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv5 = offset_convolution(ch_in=128, ch_out=64)

        self.Up4 = up_conv(ch_in=64, ch_out=32)
        self.Up_conv4 = offset_convolution(ch_in=64, ch_out=32)

        self.Up3 = up_conv(ch_in=32, ch_out=16)
        self.Up_conv3 = offset_convolution(ch_in=32, ch_out=16)

        self.Up2 = up_conv(ch_in=16, ch_out=8)
        self.Up_conv2 = offset_convolution(ch_in=16, ch_out=8)

        self.Conv_1x1 = nn.Conv2d(8, output_ch, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.Conv_1x1_1 = nn.Conv2d(8, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x, train_flag=False):
        # multi_scale_generator
        x_pre_1 = self.multi_scale_1(x)
        x_pre_2 = self.multi_scale_2(x)
        x_pre_3 = self.multi_scale_3(x)
        x_pre_4 = self.multi_scale_4(x)
        x_pre_5 = self.multi_scale_5(x)
        muti_scale_x = torch.cat((x_pre_1, x_pre_2, x_pre_3, x_pre_4, x_pre_5), dim=1)

        # encoding path
        x1 = self.Conv1(muti_scale_x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # offset convolution
        o1 = self.o1(x1)
        o2 = self.o2(x2)
        o3 = self.o3(x3)
        o4 = self.o4(x4)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((o4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((o3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((o2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((o1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        if train_flag:
            return d1
        else:
            return self.sigmoid(d1)


class MDOAU2_net_3(nn.Module):
    # delete attention block
    # offset_convolution block take place conv_block
    # skip connection
    def __init__(self, img_ch=1, output_ch=1):
        super(MDOAU2_net_3, self).__init__()
        # offset_convolution()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.multi_scale_1 = multi_scaled_dilation_conv_block(img_ch, 16, kernel_size=3, dilation=1)
        self.multi_scale_2 = multi_scaled_dilation_conv_block(img_ch, 16, kernel_size=5, dilation=1)
        self.multi_scale_3 = multi_scaled_dilation_conv_block(img_ch, 16, kernel_size=7, dilation=2)
        self.multi_scale_4 = multi_scaled_dilation_conv_block(img_ch, 16, kernel_size=9, dilation=2)
        self.multi_scale_5 = multi_scaled_dilation_conv_block(img_ch, 16, kernel_size=11, dilation=3)

        self.Conv1 = offset_convolution(ch_in=16 * 5, ch_out=8)
        self.Conv2 = offset_convolution(ch_in=8, ch_out=16)
        self.Conv3 = offset_convolution(ch_in=16, ch_out=32)
        self.Conv4 = offset_convolution(ch_in=32, ch_out=64)
        self.Conv5 = offset_convolution(ch_in=64, ch_out=128)

        self.Up5 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv5 = offset_convolution(ch_in=128, ch_out=64)

        self.Up4 = up_conv(ch_in=64, ch_out=32)
        self.Up_conv4 = offset_convolution(ch_in=64, ch_out=32)

        self.Up3 = up_conv(ch_in=32, ch_out=16)
        self.Up_conv3 = offset_convolution(ch_in=32, ch_out=16)

        self.Up2 = up_conv(ch_in=16, ch_out=8)
        self.Up_conv2 = offset_convolution(ch_in=16, ch_out=8)

        self.Conv_1x1 = nn.Conv2d(8, output_ch, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.Conv_1x1_1 = nn.Conv2d(8, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x, train_flag=False):
        # multi_scale_generator
        x_pre_1 = self.multi_scale_1(x)
        x_pre_2 = self.multi_scale_2(x)
        x_pre_3 = self.multi_scale_3(x)
        x_pre_4 = self.multi_scale_4(x)
        x_pre_5 = self.multi_scale_5(x)
        muti_scale_x = torch.cat((x_pre_1, x_pre_2, x_pre_3, x_pre_4, x_pre_5), dim=1)

        # encoding path
        x1 = self.Conv1(muti_scale_x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        if train_flag:
            return d1
        else:
            return self.sigmoid(d1)


class MDOAU2_net_4(nn.Module):
    # offset U-net++
    def __init__(self, input_channels=1, num_classes=1, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = offset_convolution(input_channels, nb_filter[0])
        self.conv1_0 = offset_convolution(nb_filter[0], nb_filter[1])
        self.conv2_0 = offset_convolution(nb_filter[1], nb_filter[2])
        self.conv3_0 = offset_convolution(nb_filter[2], nb_filter[3])
        self.conv4_0 = offset_convolution(nb_filter[3], nb_filter[4])

        self.conv0_1 = offset_convolution(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = offset_convolution(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = offset_convolution(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = offset_convolution(nb_filter[3] + nb_filter[4], nb_filter[3])

        self.conv0_2 = offset_convolution(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = offset_convolution(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = offset_convolution(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])

        self.conv0_3 = offset_convolution(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = offset_convolution(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])

        self.conv0_4 = offset_convolution(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input, train_flag=False):
        x0_0 = self.conv0_0(input)

        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            if train_flag:
                return output
            else:
                return self.sigmoid(output)

# inputs = torch.rand([1, 3, 128, 128])
# model = MDOAU2_net_4(3, 1)
# outputs = model(inputs)
# print(outputs.shape, outputs)
