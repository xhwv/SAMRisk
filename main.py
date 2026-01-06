import argparse
import os
import random
import time
import setproctitle
from models.EC.RGM import *
from models.EC.FR import *
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
from lifelines.statistics import logrank_test
from segment_anything.build_sam import sam_model_registry3D
import logging
import torch
from data.process import process
from torch.utils.data import DataLoader
from torch import nn
from lifelines.utils import concordance_index
from utils.click_method import get_next_click3D_torch_ritm, get_next_click3D_torch_2
from ranger import Ranger



logger = logging.getLogger(f"train")
local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

parser = argparse.ArgumentParser()

# Basic Information
parser.add_argument('--user', default='name of user', type=str)

parser.add_argument('--experiment', default='UCSF', type=str)

parser.add_argument('--root', default='', type=str)

parser.add_argument('--train_dir', default='train', type=str)

parser.add_argument('--valid_dir', default='val', type=str)

parser.add_argument('--mode', default='train', type=str)

parser.add_argument('--train_file', default='train.txt', type=str)

parser.add_argument('--valid_file', default='val.txt', type=str)

parser.add_argument('--dataset', default='UCSF-PDGM', type=str)

parser.add_argument('--model_name', default='UCSF', type=str)

# Training Information
parser.add_argument('--lr', default=0.0004, type=float)

parser.add_argument('--weight_decay', default=1e-5, type=float)

parser.add_argument('--amsgrad', default=True, type=bool)

parser.add_argument('--num_class', default=1, type=int)

parser.add_argument('--seed', default=1000, type=int)

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--gpu', default='0', type=str)

parser.add_argument('--num_workers', default=8, type=int)

parser.add_argument('--batch_size', default=8, type=int)

parser.add_argument('--start_epoch', default=0, type=int)

parser.add_argument('--end_epoch', default=2000, type=int)

parser.add_argument('--resume', default='', type=str)

parser.add_argument('--load', default=True, type=bool)

parser.add_argument('--point_method', default='default', type=str)

parser.add_argument('--num_clicks', type=int, default=5)

parser.add_argument('--crop_size', type=int, default=128)

parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

args = parser.parse_args()
click_methods = {
    'default': get_next_click3D_torch_2,
    'ritm': get_next_click3D_torch_2,
    'random': get_next_click3D_torch_2,
}
def CoxLoss(hazard_pred,survtime, censor):
    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i,j] = survtime[j] >= survtime[i]
    R_mat = torch.FloatTensor(R_mat).to(censor.device)#censor.device
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)
    return loss_cox

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_points(prev_masks, gt3D, click_points, click_labels):
    batch_points, batch_labels = click_methods['default'](prev_masks, gt3D)

    points_co = torch.cat(batch_points, dim=0).cuda(args.local_rank, non_blocking=True)
    points_la = torch.cat(batch_labels, dim=0).cuda(args.local_rank, non_blocking=True)

    click_points.append(points_co)
    click_labels.append(points_la)


    points_multi = torch.cat(click_points, dim=1).cuda(args.local_rank, non_blocking=True)
    labels_multi = torch.cat(click_labels, dim=1).cuda(args.local_rank, non_blocking=True)

    points_input = points_multi
    labels_input = labels_multi

    return points_input, labels_input, click_points, click_labels


def batch_forward(sam_model, image_embedding, gt3D, low_res_masks, points=None):
    sparse_embeddings, dense_embeddings = sam_model['pe'](
        points=points,
        boxes=None,
        masks=low_res_masks,
    )
    low_res_masks, iou_predictions = sam_model['de'](
        image_embeddings=image_embedding.cuda(args.local_rank, non_blocking=True),  # (B, 256, 64, 64)
        image_pe=sam_model['pe'].get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )
    prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)
    return low_res_masks, prev_masks
  
def interaction(sam_model, image_embedding, gt3D, num_clicks,click_points,click_labels):
        seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        return_loss = 0
        prev_masks = torch.zeros_like(gt3D)
        low_res_masks = F.interpolate(prev_masks.float(), size=(128//4,128//4,128//4))
        random_insert = np.random.randint(2, 9)
        for num_click in range(num_clicks):
            points_input, labels_input,click_points,click_labels = get_points(prev_masks, gt3D,click_points,click_labels)

            if num_click == random_insert or num_click == num_clicks - 1:
                low_res_masks, prev_masks = batch_forward(sam_model, image_embedding, gt3D, low_res_masks, points=None)
            else:
                low_res_masks, prev_masks = batch_forward(sam_model, image_embedding, gt3D, low_res_masks, points=[points_input, labels_input])
            loss = seg_loss(prev_masks, gt3D)
            return_loss += loss
        return prev_masks, return_loss,click_points,click_labels



def finetune_model_predict3D(img3D, gt3D, sam_model_tune, num_clicks=10
                             ,click_points=None,click_labels=None):

    img3D = sam_model_tune['emb'](img3D)
    with torch.no_grad():
        image_embedding = sam_model_tune['en'](img3D)
    prev_masks, loss,click_points_batch,click_labels_batch = interaction(sam_model_tune, image_embedding, gt3D, num_clicks=num_clicks,click_points=click_points,click_labels=click_labels)
    print_dice = get_dice_score(prev_masks, gt3D)
    print_iou=get_iou_score(prev_masks,gt3D)

    return prev_masks, click_points_batch, click_labels_batch, print_iou, print_dice, loss
                               
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))  # 可训练参数
    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

def estimate_risk(age):
    if age >= 0 and age <= 9:
        return 1
    elif age >= 10 and age <= 19:
        return 1
    elif age >= 20 and age <= 29:
        return 0.91
    elif age >= 30 and age <= 39:
        return 1.12
    elif age >= 40 and age <= 49:
        return 1.71
    elif age >= 50 and age <= 59:
        return 2.41
    elif age >= 60 and age <= 69:
        return 3.27
    elif age >= 70 and age <= 79:
        return 5.18
    elif age >= 80 :
        return 8.44

swish1=Swish()
swish2=Swish()

def lggloss(hazard_pred, clinical, survtime, censor):
    current_batch_len = len(survtime)
    rank_loss_1 = 0
    n=0
    for i in range(current_batch_len):
        for j in range(i + 1, current_batch_len):
            agei=clinical[i][1]
            agej=clinical[j][1]
            if agei<40 and agej>=40:
                if  clinical[i][2]==0 and clinical[j][2] ==0:
                        n=n+1
                        riski=estimate_risk(agei)
                        riskj=estimate_risk(agej)
                        alpha1=riskj-riski
                        alpha1=swish1(alpha1/8)
                        rank_loss_1 += torch.log(1 + torch.exp((alpha1 * (hazard_pred[i] - hazard_pred[j]))))
    if n > 0:
        rank_loss_1 = rank_loss_1 / n
    return rank_loss_1 if n >= 1 else 0

def hggloss(hazard_pred, clinical, survtime, censor):
    current_batch_len = len(survtime)
    rank_loss_1 = 0
    n=0
    for i in range(current_batch_len):
        for j in range(i + 1, current_batch_len):
            agei=clinical[i][1]
            agej=clinical[j][1]
            if agei<65 and agej>=65:
                if (clinical[i][2] in [1, 2]) and (clinical[j][2] in [1, 2]):
                        n=n+1
                        riski = estimate_risk(agei)
                        riskj = estimate_risk(agej)
                        alpha1 = riskj - riski
                        alpha1=swish2(alpha1/8)
                        rank_loss_1 += torch.log(1 + torch.exp(alpha1 * (hazard_pred[i] - hazard_pred[j])))

    if n>0:
      rank_loss_1 = rank_loss_1 / n
    return rank_loss_1 if n >= 1 else 0


class MultiTaskLossWrapper_OS(nn.Module):
    def __init__(self, loss_fn):
        super(MultiTaskLossWrapper_OS, self).__init__()
        self.loss_fn = loss_fn
        self.vars = nn.Parameter(torch.tensor((1.0,1.0,1.0),requires_grad=True)) #1.0, 6.0
    def forward(self, outputs,targets):
        loss3d = self.loss_fn[0](outputs[0], targets[0],targets[1])
        lossd_1 = torch.sum(0.5 * loss3d / (self.vars[0] ** 2) + torch.log(self.vars[0]), -1)
        loss1d=self.loss_fn[0](outputs[1], targets[0],targets[1])
        lggloss=self.loss_fn[2](outputs[0], targets[2], targets[0], targets[1])
        hggloss=self.loss_fn[3](outputs[0], targets[2], targets[0], targets[1])
        losscli=lggloss+hggloss
        losscli_1=torch.sum(0.5 * losscli / (self.vars[3] ** 2) + torch.log(self.vars[3]), -1)
        lossd_1d_1 = torch.sum(0.5 * loss1d / (self.vars[2] ** 2) + torch.log(self.vars[2]), -1)
        loss = torch.mean(lossd_1+lossd_1d_1+losscli_1)
        return loss

def main_worker():
    if args.local_rank == 0:
        log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', args.experiment + args.date)
        log_file = log_dir + '.txt'
        log_args(log_file)
        logging.info('--------------------------------------This is all argsurations----------------------------------')
        for arg in vars(args):
            logging.info('{}={}'.format(arg, getattr(args, arg)))
        logging.info('----------------------------------------This is a halving line----------------------------------')
        logging.info('{}'.format(args.description))

    setup_seed(1000)
    torch.cuda.set_device(args.local_rank)
    sam_model_tune = sam_model_registry3D["vit_b_ori"](checkpoint=None)
    model_dict = torch.load('/home/lixinyu/fenkai/sam_med3d.pth')
    state_dict = model_dict['model_state_dict']

    original_weight = state_dict['image_encoder.patch_embed.proj.weight']

    # 复制并平均权重
    with torch.no_grad():
        new_weight = original_weight.repeat(1, 4, 1, 1, 1)
    # 更新状态字典
    state_dict['image_encoder.patch_embed.proj.weight'] = new_weight
    sam_model_tune.load_state_dict(state_dict,strict=False)
    img_encoder = sam_model_tune.image_encoder
    img_encoder.load_state_dict(state_dict, strict=False)
    emb = sam_model_tune.image_emb
    model3d = get_mednet()
    RGM=Enhance1d()
    clisupervision=Enhance3DSupervision()
    criterion1 = CoxLoss
    criterion2=lggloss
    criterion3=hggloss
    MTL = MultiTaskLossWrapper_OS(loss_fn=[criterion1, criterion2,criterion3])

    for param in img_encoder.parameters():
        param.requires_grad = False

    # 然后，遍历模型中的所有Block3D模块
    for block in img_encoder.blocks:
        # 在每个Block3D中，将特定层的requires_grad设置为True
        if hasattr(block, 'tconv1'):
            for param in block.tconv1.parameters():
                param.requires_grad = True
        if hasattr(block, 'tconv2'):
            for param in block.tconv2.parameters():
                param.requires_grad = True

        if hasattr(block, 'bn1'):
            for param in block.bn1.parameters():
                param.requires_grad = True

        if hasattr(block, 'db'):
            for param in block.sia.parameters():
                param.requires_grad = True

        if hasattr(block, 'bat1'):
            for param in block.bat1.parameters():
                param.requires_grad = True

        if hasattr(block, 'bat2'):
            for param in block.bat2.parameters():
                param.requires_grad = True

        if hasattr(block, 'ln'):
            for param in block.ln.parameters():
                param.requires_grad = True

        if hasattr(block, 'l'):
            for param in block.l.parameters():
                param.requires_grad = True

        if hasattr(block, 'fc1'):
            for param in block.fc11.parameters():
                param.requires_grad = True
        if hasattr(block, 'fc2'):
            for param in block.fc22.parameters():
                param.requires_grad = True
        if hasattr(block, 'attn'):
            for param in block.attn.factor1.parameters():
                param.requires_grad = True
            for param in block.attn.factor2.parameters():
                param.requires_grad = True


    for name, param in img_encoder.named_parameters():
        if 'patch_embed' in name:
            # if 'neck' in name or 'patch_embed' in name:
            param.requires_grad = True

    nets = {
        'en': img_encoder.cuda(),
        'model3d': model3d.cuda(),
        'mtl': MTL.cuda(),
        'emb':emb.cuda(),
        'enhance':RGM.cuda(),
        'encli':clisupervision.cuda()
    }

    param = [p for v in nets.values() for p in list(v.parameters()) if
             p.requires_grad] # Only parameters that require gradients
    total_trainable_params = sum(p.numel() for p in param)
    print(total_trainable_params)
    optimizer = Ranger(
        param,  # 网络的可训练参数
        lr=args.lr,  # 学习率
        weight_decay=args.weight_decay  # 权重衰减
    )

    if args.local_rank == 0:
        roott=""
        checkpoint_dir = os.path.join(roott, 'checkpoint', args.experiment+args.date)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    train_list = os.path.join(args.root, args.train_dir, args.train_file)
    train_root = os.path.join(args.root, args.train_dir)
    train_set = process(train_list, train_root, args.mode)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    logging.info('Samples for train = {}'.format(len(train_set)))
    num_gpu = (len(args.gpu)+1) // 2

    train_loader = DataLoader(dataset=train_set, shuffle=False, batch_size=args.batch_size // num_gpu,
                              drop_last=True, num_workers=args.num_workers, pin_memory=True)
    valid_list = os.path.join(args.root, args.valid_dir, args.valid_file)
    valid_root = os.path.join(args.root, args.valid_dir)
    valid_set = process(valid_list, valid_root, 'valid')
    valid_loader = DataLoader(valid_set, batch_size=2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    torch.set_grad_enabled(True)
    es=0
    torch.cuda.empty_cache()
    epochs_no_improve = 0


    for epoch in range(args.start_epoch, args.end_epoch):
        es += 1
        # 每个epoch开始时调整一次学习率
        current_lr = adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)
        print(f"Epoch {epoch} / {args.end_epoch - 1}")
        setproctitle.setproctitle('{}: {}/{}'.format(args.user, epoch + 1, args.end_epoch))
        print(f"LR = {current_lr}")
        # 设为训练模式
        nets['en'].train()
        nets['model3d'].train()
        nets['mtl'].train()
        nets['emb'].train()
        nets['enhance'].train()
        nets['encli'].train()
        predicted_score = []
        predicted_1d = []
        true_event_time = []
        true_statuse = []
        names = []
        scores = []
        optimizer.zero_grad()
        train_loss_sum = 0.0
        num_batches = 0
        for i, data in enumerate(train_loader):
            x, time, status, clinical, name = data
            x = x.cuda(args.local_rank, non_blocking=True)
            time = time.cuda(args.local_rank, non_blocking=True)
            status = status.cuda(args.local_rank, non_blocking=True)
            clinical = clinical.cuda(args.local_rank, non_blocking=True)

            # 前向
            enhance = nets['enhance'](clinical)
            output1d = nets['encli'](enhance)
            img3D = nets['emb'](x, enhance)
            image_embedding = nets['en'](img3D)
            output3d = nets['model3d'](image_embedding)
            names.extend(name)
            s3d = output3d.detach().cpu().numpy()
            scores.extend(s3d.tolist())
            predicted_score += list(s3d.flatten())
            predicted_1d += list(output1d.detach().cpu().numpy().flatten())
            true_event_time += list(time.detach().cpu().numpy())
            true_statuse += list(status.detach().cpu().numpy())

            # 损失
            loss = nets['mtl']([output3d, output1d], [time, status, clinical])

            # 反向传播
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # 聚合训练loss
            train_loss_sum += loss.item()
            num_batches += 1

        # 平均训练损失
        train_loss = train_loss_sum / max(1, num_batches)

        print(predicted_score)
        print('接下来是1d的预测结果')
        print(predicted_1d)

        tindex = concordance_index(np.array(true_event_time),
                                   -np.array(predicted_score),
                                   np.array(true_statuse))
        pvalue = cox_log_rank(np.array(predicted_score),
                              np.array(true_event_time),
                              np.array(true_statuse))

        print("第" + str(es) + "个epoch的" + "loss is " + str(train_loss))
        print("第" + str(es) + "个epoch的" + "c-index is " + str(tindex))
        print("第" + str(es) + "个epoch的" + "pvalue is " + str(pvalue))

        # ===== 验证 =====
        predicted_score_val = []
        true_event_time_val = []
        true_status_val = []

        with torch.no_grad():
            nets['en'].eval()
            nets['model3d'].eval()
            nets['enhance'].eval()
            nets['mtl'].eval()
            nets['emb'].eval()
            nets['encli'].eval()

            for i, data in enumerate(valid_loader):
                x, time, status, clinical, name = data
                x = x.cuda(args.local_rank, non_blocking=True)
                time = time.cuda(args.local_rank, non_blocking=True)
                status = status.cuda(args.local_rank, non_blocking=True)
                clinical = clinical.cuda(args.local_rank, non_blocking=True)

                enhance = nets['enhance'](clinical)
                img3D = nets['emb'](x, enhance)
                image_embedding = nets['en'](img3D)
                output3d = nets['model3d'](image_embedding)

                predicted_score_val += list(output3d.detach().cpu().numpy().flatten())
                true_event_time_val += list(time.cpu().numpy())
                true_status_val += list(status.cpu().numpy())

            tindex_val = concordance_index(np.array(true_event_time_val),
                                           -np.array(predicted_score_val),
                                           np.array(true_status_val))
            pvalue_val = cox_log_rank(np.array(predicted_score_val),
                                      np.array(true_event_time_val),
                                      np.array(true_status_val))

            print(predicted_score_val)
            print(f"Validation C-index for epoch {es}: {tindex_val}")
            print(f"Validation P-value for epoch {es}: {pvalue_val}")

            # ======= 若验证集C-index创新高则保存 =======
            if tindex_val > best_val_cindex:
                best_val_cindex = tindex_val
                # 构造文件名并保存
                os.makedirs(checkpoint_dir, exist_ok=True)
                model_filename = f"epoch{epoch + 1:03d}_valC{tindex_val:.4f}_p{pvalue_val:.3e}.pth"
                model_path = os.path.join(checkpoint_dir, model_filename)

                torch.save({
                    'epoch': epoch,
                    'model3d': nets['model3d'].state_dict(),
                    'enhance': nets['enhance'].state_dict(),
                    'emb': nets['emb'].state_dict(),
                    'mtl': nets['mtl'].state_dict(),
                    'en': nets['en'].state_dict(),
                    'encli': nets['encli'].state_dict(),
                    'val_cindex': tindex_val,
                    'val_pvalue': pvalue_val,
                    'train_loss': train_loss,
                    'lr': current_lr,
                    'args': vars(args) if hasattr(args, "__dict__") else None,
                }, model_path)

                print(
                    f"[BEST so far] Model saved to {model_path} (best val C-index: {best_val_cindex:.4f} @ epoch {epoch + 1})")

            if train_loss < best_train_loss - 1e-12:
                best_train_loss = train_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= 100:
                print(f"Early stopping: training loss has not improved for {epochs_no_improve} epochs. "
                      f"Best train loss: {best_train_loss:.6f}, last epoch: {epoch + 1}")
                break



from sklearn.utils import resample
import numpy as np
def bootstrap_c_index(true_event_times, predicted_scores, true_statuses, n_bootstrap=1000):
    c_indexes = []
    for _ in range(n_bootstrap):
        indices = resample(np.arange(len(true_event_times))) # 生成bootstrap样本的索引
        bs_event_times = true_event_times[indices]
        bs_predicted_scores = predicted_scores[indices]
        bs_statuses = true_statuses[indices]

        c_index_value = concordance_index(bs_event_times, -bs_predicted_scores, bs_statuses)
        c_indexes.append(c_index_value)

    # 计算置信区间
    c_indexes = np.array(c_indexes)

    lower_bound = np.percentile(c_indexes, 2.5)  # 2.5%分位数
    upper_bound = np.percentile(c_indexes, 97.5)  # 97.5%分位数

    return lower_bound, upper_bound




def cox_log_rank(hazardsdata,survtime_all,labels):
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return(pvalue_pred)
def log_args(log_file):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main_worker()
