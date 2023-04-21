import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
import warnings
def get_loss(net_output, ground_truth, label_choose):
    loss_dict = {}
    loss = 0.0
    if 'rx' in label_choose:
        rx_loss = F.cross_entropy(net_output['rx'], ground_truth['rx'])
        loss += rx_loss
        loss_dict['rx'] = rx_loss
    if 'ry' in label_choose:
        ry_loss = F.cross_entropy(net_output['ry'], ground_truth['ry'])
        loss += ry_loss
        loss_dict['ry'] = ry_loss
    if 'rz' in label_choose:
        rz_loss = F.cross_entropy(net_output['rz'], ground_truth['rz'])
        loss += rz_loss
        loss_dict['rz'] = rz_loss
    if 'tx' in label_choose:
        tx_loss = F.cross_entropy(net_output['tx'], ground_truth['tx'])
        loss += tx_loss
        loss_dict['tx'] = tx_loss
    if 'ty' in label_choose:
        ty_loss = F.cross_entropy(net_output['ty'], ground_truth['ty'])
        loss += ty_loss
        loss_dict['ty'] = ty_loss
    if 'tz' in label_choose:
        tz_loss = F.cross_entropy(net_output['tz'], ground_truth['tz'])
        loss += tz_loss
        loss_dict['tz'] = tz_loss
    return loss, loss_dict


def calculate_metrics(output, target, label_choose):

    acc_lst = []
    with warnings.catch_warnings():  # sklearn 在处理混淆矩阵中的零行时可能会产生警告
        warnings.simplefilter("ignore")
        if 'rx' in label_choose:
            _, predicted_rx = output['rx'].cpu().max(1)
            gt_rx = target['rx'].cpu()
            acc_rx = balanced_accuracy_score(y_true=gt_rx.numpy(), y_pred=predicted_rx.numpy())
            acc_lst.append(acc_rx)
        if 'ry' in label_choose:
            _, predicted_ry = output['ry'].cpu().max(1)
            gt_ry = target['ry'].cpu()
            acc_ry = balanced_accuracy_score(y_true=gt_ry.numpy(), y_pred=predicted_ry.numpy())
            acc_lst.append(acc_ry)
        if 'rz' in label_choose:
            _, predicted_rz = output['rz'].cpu().max(1)
            gt_rz = target['rz'].cpu()
            acc_rz = balanced_accuracy_score(y_true=gt_rz.numpy(), y_pred=predicted_rz.numpy())
            acc_lst.append(acc_rz)
        if 'tx' in label_choose:
            _, predicted_tx = output['tx'].cpu().max(1)
            gt_tx = target['tx'].cpu()
            acc_tx = balanced_accuracy_score(y_true=gt_tx.numpy(), y_pred=predicted_tx.numpy())
            acc_lst.append(acc_tx)
        if 'ty' in label_choose:
            _, predicted_ty = output['ty'].cpu().max(1)
            gt_ty = target['ty'].cpu()
            acc_ty = balanced_accuracy_score(y_true=gt_ty.numpy(), y_pred=predicted_ty.numpy())
            acc_lst.append(acc_ty)
        if 'tz' in label_choose:
            _, predicted_tz = output['tz'].cpu().max(1)
            gt_tz = target['tz'].cpu()
            acc_tz = balanced_accuracy_score(y_true=gt_tz.numpy(), y_pred=predicted_tz.numpy())
            acc_lst.append(acc_tz)

    return acc_lst