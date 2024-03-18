import torch.nn as nn
import torch.nn.functional as F
import torch
from util.sobel import SobelComputer



"""BCE loss"""


class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss(weight=weight, size_average=size_average)

    def forward(self, pred, target):
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        loss = self.bceloss(pred_flat, target_flat)

        return loss


"""Dice loss"""


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1

        size = pred.size(0)

        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        intersection = pred_flat * target_flat
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss


"""BCE + DICE Loss"""


class BceDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss(weight, size_average)
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)
        loss = diceloss + bceloss

        return loss

class BceDiceLoss_lapulasi(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BceDiceLoss_lapulasi, self).__init__()
        self.bce = BCELoss(weight, size_average)
        self.dice = DiceLoss()


    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)
        loss = bceloss +diceloss

        return loss


""" Deep Supervision Loss"""
def get_hard_pred(pred,hard_index):
    hard_preds = pred[hard_index[0]].unsqueeze(0)
    for j in hard_index[1::]:
        hard_pred = pred[j].unsqueeze(0)
        hard_preds = torch.cat([hard_preds,hard_pred],dim=0)   
    return hard_preds



def DeepSupervisionLoss(output, targets,hard_index):
    d0, d1, d2, d3, d4,d51,d41,d31,d21,d11 = output[0:]
    ##此处需要修改
    masks = targets[0]['mask']
    for bs in range(1,len(targets)):
        mask = targets[bs]['mask']
        if len(masks.shape)==4:
            mask= torch.unsqueeze(mask,0)
            masks = torch.cat([masks,mask],dim=0)
            continue
        masks = torch.stack([masks,mask],dim=0)  
    if masks.shape[0] ==1:
        masks = masks.unsqueeze(0)

#将有注释的标签提取出来
    gts = masks[hard_index[0]].unsqueeze(0)
    for i in hard_index[1::]:
        gt = masks[i].unsqueeze(0)
        gts = torch.cat([gts,gt],dim=0)


#将有注释的对应的预测提取出来
    d0 = get_hard_pred(d0,hard_index)
    d1 = get_hard_pred(d1,hard_index)
    d2 = get_hard_pred(d2,hard_index)
    d3 = get_hard_pred(d3,hard_index)
    d4 = get_hard_pred(d4,hard_index)

    criterion = BceDiceLoss()

    loss0 = criterion(d0, gts)
    gts = F.interpolate(gts, scale_factor=0.5, mode='bilinear', align_corners=True)

    loss1 = criterion(d1, gts)
    gts = F.interpolate(gts, scale_factor=0.5, mode='bilinear', align_corners=True)

    loss2 = criterion(d2, gts)
    gts = F.interpolate(gts, scale_factor=0.5, mode='bilinear', align_corners=True)

    loss3 = criterion(d3, gts)
    gts = F.interpolate(gts, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss4 = criterion(d4, gts)

    return loss0 + loss1 + loss2 + loss3 + loss4

def DeepSupervisionLoss1(output, gts):
    d0, d1, d2, d3, d4,d51,d41,d31,d21,d11 = output[0:]
    gts = (gts>=0.5).float()


    # criterion_kd = nn.KLDivLoss()
    # d0 = torch.log(d0.flatten(1))
    # d1 = torch.log(d1.flatten(1))
    # d2 = torch.log(d2.flatten(1))
    # d3 = torch.log(d3.flatten(1))
    # d4 = torch.log(d4.flatten(1))

    criterion = BceDiceLoss()

    loss0 = criterion(d0, gts)
    gts = F.interpolate(gts, scale_factor=0.5, mode='bilinear', align_corners=True)

    loss1 = criterion(d1, gts)
    gts = F.interpolate(gts, scale_factor=0.5, mode='bilinear', align_corners=True)

    loss2 = criterion(d2, gts)
    gts = F.interpolate(gts, scale_factor=0.5, mode='bilinear', align_corners=True)

    loss3 = criterion(d3, gts)
    gts = F.interpolate(gts, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss4 = criterion(d4, gts)

    return loss0 + loss1 + loss2 + loss3 + loss4

