import torch.nn as nn
import torch
import torch.nn.functional as F
from .DeepLabV3 import DeepLabV3

class TVSD(nn.Module):
    def __init__(self, pretrained_path=None, num_classes=1, all_channel=256, all_dim=26 * 26, T=0.07):  # 473./8=60 416./8=52
        super(TVSD, self).__init__()
        self.encoder = DeepLabV3()
        self.T = T
        # load pretrained model from DeepLabV3 module
        # in our experiments, no need to pretrain the single deeplabv3
        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path)
            print(f"Load checkpoint:{pretrained_path}")
            self.encoder.load_state_dict(checkpoint['model'])
        self.co_attention = CoattentionModel(num_classes=num_classes, all_channel=all_channel, all_dim=all_dim)
        self.project = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.final_pre = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        initialize_weights(self.co_attention, self.project, self.final_pre)

    def forward(self, input1, input2, input3):
        input_size = input1.size()[2:]
        low_exemplar, exemplar, _ = self.encoder(input1)
        low_query, query, _ = self.encoder(input2)
        low_other, other, _ = self.encoder(input3)
        x1, x2 = self.co_attention(exemplar, query)
        x1 = F.interpolate(x1, size=low_exemplar.shape[2:], mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, size=low_query.shape[2:], mode='bilinear', align_corners=False)
        x3 = F.interpolate(other, size=low_other.shape[2:], mode='bilinear', align_corners=False)
        fuse_exemplar = torch.cat([x1, self.project(low_exemplar)], dim=1)
        fuse_query = torch.cat([x2, self.project(low_query)], dim=1)
        fuse_other = torch.cat([x3, self.project(low_other)], dim=1)
        exemplar_pre = self.final_pre(fuse_exemplar)
        query_pre = self.final_pre(fuse_query)
        other_pre = self.final_pre(fuse_other)

        # scene vector
        v1 = F.adaptive_avg_pool2d(exemplar, (1, 1)).squeeze(-1).squeeze(-1)
        v1 = nn.functional.normalize(v1, dim=1)
        v2 = F.adaptive_avg_pool2d(query, (1, 1)).squeeze(-1).squeeze(-1)
        v2 = nn.functional.normalize(v2, dim=1)
        v3 = F.adaptive_avg_pool2d(other, (1, 1)).squeeze(-1).squeeze(-1)
        v3 = nn.functional.normalize(v3, dim=1)

        l_pos = torch.einsum('nc,nc->n', [v1, v2]).unsqueeze(-1)
        l_neg1 = torch.einsum('nc,nc->n', [v1, v3]).unsqueeze(-1)
        # l_neg2 = torch.einsum('nc,nc->n', [v2, v3]).unsqueeze(-1)
        # logits = torch.cat([l_pos, l_neg1, l_neg2], dim=1)
        logits = torch.cat([l_pos, l_neg1], dim=1)
        logits /= self.T
        exemplar_pre = F.upsample(exemplar_pre, input_size, mode='bilinear', align_corners=False)  # upsample to the size of input image, scale=8
        query_pre = F.upsample(query_pre, input_size, mode='bilinear', align_corners=False)  # upsample to the size of input image, scale=8
        other_pre = F.upsample(other_pre, input_size, mode='bilinear', align_corners=False)  # upsample to the size of input image, scale=8
        return exemplar_pre, query_pre, other_pre, logits



class CoattentionModel(nn.Module):  # spatial and channel attention module
    def __init__(self, num_classes=1, all_channel=256, all_dim=26 * 26):  # 473./8=60 416./8=52
        super(CoattentionModel, self).__init__()
        self.linear_e = nn.Linear(all_channel, all_channel, bias=False)
        self.channel = all_channel
        self.dim = all_dim
        self.gate1 = nn.Conv2d(all_channel * 2, 1, kernel_size=1, bias=False)
        self.gate2 = nn.Conv2d(all_channel * 2, 1, kernel_size=1, bias=False)
        self.gate_s = nn.Sigmoid()
        self.conv1 = nn.Conv2d(all_channel * 2, all_channel, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(all_channel * 2, all_channel, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(all_channel)
        self.bn2 = nn.BatchNorm2d(all_channel)
        self.prelu = nn.ReLU(inplace=True)
        self.globalAvgPool = nn.AvgPool2d(26, stride=1)
        self.fc1 = nn.Linear(in_features=256*2, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=256)
        self.fc3 = nn.Linear(in_features=256*2, out_features=16)
        self.fc4 = nn.Linear(in_features=16, out_features=256)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, exemplar, query):

        # spatial co-attention
        fea_size = query.size()[2:]
        all_dim = fea_size[0] * fea_size[1]
        exemplar_flat = exemplar.view(-1, query.size()[1], all_dim)  # N,C,H*W
        query_flat = query.view(-1, query.size()[1], all_dim)
        exemplar_t = torch.transpose(exemplar_flat, 1, 2).contiguous()  # batch size x dim x num
        exemplar_corr = self.linear_e(exemplar_t)  #
        A = torch.bmm(exemplar_corr, query_flat)
        A1 = F.softmax(A.clone(), dim=1)  #
        B = F.softmax(torch.transpose(A, 1, 2), dim=1)
        query_att = torch.bmm(exemplar_flat, A1).contiguous()
        exemplar_att = torch.bmm(query_flat, B).contiguous()
        input1_att = exemplar_att.view(-1, query.size()[1], fea_size[0], fea_size[1])
        input2_att = query_att.view(-1, query.size()[1], fea_size[0], fea_size[1])
        
        # spacial attention
        input1_mask = self.gate1(torch.cat([input1_att, input2_att], dim=1))
        input2_mask = self.gate2(torch.cat([input1_att, input2_att], dim=1))
        input1_mask = self.gate_s(input1_mask)
        input2_mask = self.gate_s(input2_mask)

        # channel attention
        out_e = self.globalAvgPool(torch.cat([input1_att, input2_att], dim=1))
        out_e = out_e.view(out_e.size(0), -1)
        out_e = self.fc1(out_e)
        out_e = self.relu(out_e)
        out_e = self.fc2(out_e)
        out_e = self.sigmoid(out_e)
        out_e = out_e.view(out_e.size(0), out_e.size(1), 1, 1)
        out_q = self.globalAvgPool(torch.cat([input1_att, input2_att], dim=1))
        out_q = out_q.view(out_q.size(0), -1)
        out_q = self.fc3(out_q)
        out_q = self.relu(out_q)
        out_q = self.fc4(out_q)
        out_q = self.sigmoid(out_q)
        out_q = out_q.view(out_q.size(0), out_q.size(1), 1, 1)

        # apply dual attention masks
        input1_att = input1_att * input1_mask
        input2_att = input2_att * input2_mask
        input2_att = out_e * input2_att
        input1_att = out_q * input1_att

        # concate original feature
        input1_att = torch.cat([input1_att, exemplar], 1)
        input2_att = torch.cat([input2_att, query], 1)
        input1_att = self.conv1(input1_att)
        input2_att = self.conv2(input2_att)
        input1_att = self.bn1(input1_att)
        input2_att = self.bn2(input2_att)
        input1_att = self.prelu(input1_att)
        input2_att = self.prelu(input2_att)

        return input1_att, input2_att  # shape: NxCx

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
