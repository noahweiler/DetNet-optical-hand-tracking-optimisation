# === FINETUNE-PRUNED CHANGE (d): suppress Windows OMP Error #15 ===
# numpy (MKL) ships libiomp5md.dll, torch ships libomp.dll; whichever OMP
# runtime loads second aborts the process (FATAL) unless this is set. It must
# be set BEFORE any import that pulls in an OpenMP-linked library (cv2 / numpy
# / torch). On Windows the DataLoader spawns workers that re-import this
# module, so this top-of-file placement (plus os.environ inheritance at spawn)
# covers every worker process too — without it, each worker dies and the
# DataLoader fails with "worker(s) exited unexpectedly".
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ===========================================================================

# === FINETUNE-PRUNED CHANGE (c): import cv2 BEFORE torch ===
# On Windows there is a DLL load-order conflict in this env: if torch / the
# datasets import chain loads first, cv2's native .pyd then fails with
# "DLL load failed while importing cv2". Importing cv2 first avoids it.
import cv2  # noqa: F401  -- import-for-side-effect; keep above torch
# === FINETUNE-PRUNED CHANGE (c): make old_training/ importable ===
# losses/ and datasets/ live under old_training/ in this repo layout.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'old_training'))
# ===========================================================================

import argparse
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as torch_f  # === FINETUNE-PRUNED CHANGE (a): needed by DetLoss2D ===
from progress.bar import Bar
from tqdm import tqdm

import losses as losses
import utils.misc as misc
from datasets.egodexter import EgoDexter
from datasets.handataset import HandDataset
from model.detnet import detnet
from utils import func, align
from utils.eval.evalutils import AverageMeter, accuracy_heatmap
from utils.eval.zimeval import EvalUtil

# select proper device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
DEBUG = 0


# === FINETUNE-PRUNED CHANGE (a): 2D-only loss ===============================
# The 2D-only detnet produces only {'h_map', 'uv'} (the 3D dmap/lmap heads are
# commented out), so losses.DetLoss.compute_loss crashes at preds['d_map'][flag].
# DetLoss2D reproduces DetLoss.compute_loss's HEATMAP term verbatim
# (lambda_hm * sum_j 0.5 * mse_loss(pred_hm_j * hm_veil_j, targ_hm_j * hm_veil_j))
# and drops the dm/lm terms. It returns the SAME (final_loss, det_losses,
# batch_3d_size) shape DetLoss does, with det_dm/det_lm = 0 and a dummy
# batch_3d_size = 1, so train()/one_forward_pass() need no other change.
class DetLoss2D:
    def __init__(self, lambda_hm=100):
        self.lambda_hm = lambda_hm

    def compute_loss(self, preds, targs, infos):
        hm_veil = infos['hm_veil']
        batch_size = infos['batch_size']

        final_loss = torch.Tensor([0]).cuda()
        det_losses = {}

        pred_hm = preds['h_map']
        targ_hm = targs['hm']  # B*21*32*32

        # compute hmloss anyway  (verbatim from losses.DetLoss.compute_loss)
        hm_loss = torch.Tensor([0]).cuda()
        if self.lambda_hm:
            hm_veil = hm_veil.unsqueeze(-1)
            njoints = pred_hm.size(1)
            pred_hm = pred_hm.reshape((batch_size, njoints, -1)).split(1, 1)
            targ_hm = targ_hm.reshape((batch_size, njoints, -1)).split(1, 1)
            for idx in range(njoints):
                pred_hmapi = pred_hm[idx].squeeze()  # (B, 1, 1024)->(B, 1024)
                targ_hmi = targ_hm[idx].squeeze()
                hm_loss += 0.5 * torch_f.mse_loss(
                    pred_hmapi.mul(hm_veil[:, idx]),  # (B, 1024) mul (B, 1)
                    targ_hmi.mul(hm_veil[:, idx])
                )
            final_loss += self.lambda_hm * hm_loss
        det_losses["det_hm"] = hm_loss

        # 2D-only: no delta-map / location-map terms (kept as zeros for logging)
        det_losses["det_dm"] = torch.Tensor([0]).cuda()
        det_losses["det_lm"] = torch.Tensor([0]).cuda()
        det_losses["det_total"] = final_loss

        # dummy (>=1) so train()'s am_loss_dm.update(0, n) never divides by zero
        batch_3d_size = torch.Tensor([1]).cuda()
        return final_loss, det_losses, batch_3d_size
# ===========================================================================


def main(args):
    for path in [args.checkpoint, args.outpath]:
        if not os.path.isdir(path):
            os.makedirs(path)

    misc.print_args(args)

    print("\nCREATE NETWORK")
    # === FINETUNE-PRUNED CHANGE (b): load the pruned saved-model object ======
    if args.pruned_model_path:
        print("LOADING PRUNED MODEL: {}".format(args.pruned_model_path))
        model = torch.load(args.pruned_model_path, map_location='cpu', weights_only=False)
        print("Pruned model parameters: {:,}".format(sum(p.numel() for p in model.parameters())))
    else:
        model = detnet()
    # ========================================================================
    model.to(device)

    # define loss function (criterion) and optimizer
    # === FINETUNE-PRUNED CHANGE (a): 2D-only loss ===========================
    if args.pruned_model_path:
        criterion_det = DetLoss2D(
            lambda_hm=100.,
        )
    else:
        criterion_det = losses.DetLoss(
            lambda_hm=100.,
            lambda_dm=1.,
            lambda_lm=10.,
        )
    # ========================================================================
    criterion = {
        'det': criterion_det
    }
    optimizer = torch.optim.Adam(
        [
            {
                'params': model.parameters(),
                'initial_lr': args.learning_rate
            },

        ],
        lr=args.learning_rate
    )

    test_set_dic = {}
    test_loader_dic = {}
    best_acc = {}
    auc_all = {}
    acc_hm_all = {}
    for test_set_name in args.datasets_test:
        if test_set_name in ['stb', 'rhd', 'do']:
            test_set_dic[test_set_name] = HandDataset(
                data_split='test',
                train=False,
                subset_name=test_set_name,
                data_root=args.data_root,
            )
        elif test_set_name == 'eo':
            test_set_dic[test_set_name] = EgoDexter(
                data_split='test',
                data_root=args.data_root,
                hand_side="right"
            )
            print(test_set_dic[test_set_name])

        test_loader_dic[test_set_name] = torch.utils.data.DataLoader(
            test_set_dic[test_set_name],
            batch_size=args.test_batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True, drop_last=False
        )
        best_acc[test_set_name] = 0
        auc_all[test_set_name] = []
        acc_hm_all[test_set_name] = []

    total_test_set_size = 0
    for key, value in test_set_dic.items():
        total_test_set_size += len(value)
    print("Total test set size: {}".format(total_test_set_size))

    if args.resume or args.evaluate:
        print("\nLOAD CHECKPOINT")
        state_dict = torch.load(os.path.join(
            args.checkpoint,
            'ckp_detnet_{}.pth'.format(args.evaluate_id)
        ))
        # if args.clean:
        state_dict = misc.clean_state_dict(state_dict)

        model.load_state_dict(state_dict)
    # === FINETUNE-PRUNED CHANGE (b): pruned model already loaded above; ======
    # its weights are the trained pruned weights -- do NOT re-initialise them.
    elif args.pruned_model_path:
        pass
    # ========================================================================
    else:
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

    if args.evaluate:
        for key, value in test_loader_dic.items():
            validate(value, model, criterion, key, args=args)
        return 0

    train_dataset = HandDataset(
        data_split='train',
        train=True,
        subset_name=args.datasets_train,
        data_root=args.data_root,
        scale_jittering=0.1,
        center_jettering=0.1,
        max_rot=0.5 * np.pi,
    )

    print("Total train dataset size: {}".format(len(train_dataset)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True, drop_last=False
    )

    # DataParallel so u can use multi GPUs
    model = torch.nn.DataParallel(model)
    print("\nUSING {} GPUs".format(torch.cuda.device_count()))

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_decay_step, gamma=args.gamma,
        last_epoch=args.start_epoch
    )

    acc_hm = {}
    loss_all = {"lossH": [],
                "lossD": [],
                "lossL": [],

                }

    for epoch in range(args.start_epoch, args.epochs + 1):
        print('\nEpoch: %d' % (epoch + 1))
        for i in range(len(optimizer.param_groups)):
            print('group %d lr:' % i, optimizer.param_groups[i]['lr'])
        #############  trian for one epoch  ###############
        train(
            train_loader,
            model,
            criterion,
            optimizer,
            args=args, loss_all=loss_all
        )
        ##################################################
        auc = best_acc.copy() # need to deepcopy it because it's a dict
        for key, value in test_loader_dic.items():
            auc[key], acc_hm[key] = validate(value, model, criterion, key, args=args)
            auc_all[key].append([epoch + 1, auc[key]])
            acc_hm_all[key].append([epoch + 1, acc_hm[key]])

        misc.save_checkpoint(
            {
                'epoch': epoch + 1,
                'model': model,
            },
            checkpoint=args.checkpoint,
            filename='{}.pth'.format(args.saved_prefix),
            snapshot=args.snapshot,
            is_best=[auc, best_acc]
        )

        for key, value in test_loader_dic.items():
            if auc[key] > best_acc[key]:
                best_acc[key] = auc[key]

        misc.out_loss_auc(loss_all, auc_all, acc_hm_all, outpath=args.outpath)

        scheduler.step()

    return 0  # end of main


def one_forward_pass(metas, model, criterion, args, train=True):
    clr = metas['clr'].to(device, non_blocking=True)

    ''' prepare infos '''
    if 'hm_veil' in metas.keys():
        hm_veil = metas['hm_veil'].to(device, non_blocking=True)  # (B,21)

        infos = {
            'hm_veil': hm_veil,
            'batch_size': clr.shape[0]
        }

        ''' prepare targets '''

        hm = metas['hm'].to(device, non_blocking=True)
        delta_map = metas['delta_map'].to(device, non_blocking=True)
        location_map = metas['location_map'].to(device, non_blocking=True)
        flag_3d = metas['flag_3d'].to(device, non_blocking=True)
        joint = metas['joint'].to(device, non_blocking=True)

        targets = {
            'clr': clr,
            'hm': hm,
            'dm': delta_map,
            'lm': location_map,
            "flag_3d": flag_3d,
            "joint": joint

        }
    else:
        infos = {
            'batch_size': clr.shape[0]
        }
        tips = metas['tips'].to(device, non_blocking=True)
        targets = {
            'clr': clr,
            "joint": tips

        }

    ''' ----------------  Forward Pass  ---------------- '''
    results = model(clr)
    ''' ----------------  Forward End   ---------------- '''

    total_loss = torch.Tensor([0]).cuda()
    losses = {}

    if not train:
        return results, {**targets, **infos}, total_loss, losses

    ''' compute losses '''
    if args.det_loss:
        det_total_loss, det_losses, batch_3d_size = criterion['det'].compute_loss(
            results, targets, infos
        )
        total_loss += det_total_loss
        losses.update(det_losses)

        targets["batch_3d_size"] = batch_3d_size

    return results, {**targets, **infos}, total_loss, losses


def validate(val_loader, model, criterion, key, args, stop=-1):
    print("{}_test_set under test".format(key))
    # switch to evaluate mode
    model.eval()

    if key in ["stb", "rhd"]:
        am_accH = AverageMeter()

    evaluator = EvalUtil()

    if args.evaluate:
        gt_joints = []
        pre_joints = []

    with torch.no_grad():
        for i, metas in tqdm(enumerate(val_loader)):
            preds, targets, _1, _2 = one_forward_pass(
                metas, model, criterion, args=None, train=False
            )

            if key in ["stb", "rhd"]:
                # heatmap accuracy
                avg_acc_hm, _ = accuracy_heatmap(
                    preds['h_map'],
                    targets['hm'],
                    targets['hm_veil']
                )
                am_accH.update(avg_acc_hm, targets['batch_size'])

            # === FINETUNE-PRUNED CHANGE (a): 2D model has no 'xyz' ===========
            # The 3D-joint AUC path below reads preds['xyz']; skip it for the
            # pruned 2D model and use heatmap accuracy as the per-epoch metric.
            # (The full 2D PCK evaluation is done separately by evaluate_detnet.py.)
            if args.pruned_model_path:
                if stop != -1 and i >= stop:
                    break
                continue
            # ================================================================

            pred_joint = func.to_numpy(preds['xyz'])

            gt_joint = func.to_numpy(targets['joint'])

            if args.evaluate:
                gt_joints.extend(gt_joint.tolist())
                pre_joints.extend(pred_joint.tolist())

            gt_joint, pred_joint_align = align.global_align(gt_joint, pred_joint, key=key)


            for targj, predj_a in zip(gt_joint, pred_joint_align):
                evaluator.feed(targj * 1000.0, predj_a * 1000.0)
                # vis.multi_plot3d([targj * 1000.0, predj_a * 1000.0], title=["target", "pred"])

            if stop != -1 and i >= stop:
                break

    # === FINETUNE-PRUNED CHANGE (a): report heatmap accuracy for 2D model ===
    if args.pruned_model_path:
        if key in ["stb", "rhd"]:
            print("heatmap accuracy of {}_test_set is : {}".format(key, am_accH.avg))
            return am_accH.avg, am_accH.avg
        else:
            return 0, 0
    # ========================================================================

    if args.evaluate:

        gt_joints = np.array(gt_joints)
        pre_joints = np.array(pre_joints)
        out_path = "out_testset"
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        np.save("{}/{}_gt_joints.npy".format(out_path, key), gt_joints)
        np.save("{}/{}_pre_joints.npy".format(out_path, key), pre_joints)

    (
        _1, _2, _3,
        auc_all,
        pck_curve_all,
        thresholds
    ) = evaluator.get_measures(
        20, 50, 15
    )
    print("AUC all of {}_test_set is : {}".format(key, auc_all))

    if key in ["stb", "rhd"]:
        return auc_all, am_accH.avg
    elif key in ["do", "eo"]:
        return auc_all, 0


def train(train_loader, model, criterion, optimizer, args, loss_all):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    am_loss_hm = AverageMeter()
    am_loss_dm = AverageMeter()
    am_loss_lm = AverageMeter()

    last = time.time()
    # switch to trian
    model.train()
    bar = Bar('\033[31m Train \033[0m', max=len(train_loader))
    # for i, metas in tqdm(enumerate(train_loader)):
    for i, metas in enumerate(train_loader):
        data_time.update(time.time() - last)
        results, targets, total_loss, losses = one_forward_pass(
            metas, model, criterion, args, train=True
        )

        am_loss_hm.update(losses['det_hm'].item(), targets['batch_size'])
        am_loss_dm.update(losses['det_dm'].item(), targets['batch_3d_size'].item())
        am_loss_lm.update(losses['det_lm'].item(), targets['batch_3d_size'].item())

        ''' backward and step '''
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        ''' progress '''
        batch_time.update(time.time() - last)
        last = time.time()
        bar.suffix = (
            '({batch}/{size}) '
            'd: {data:.2f}s | '
            'b: {bt:.2f}s | '
            't: {total:}s | '
            'eta:{eta:}s | '
            'lH: {lossH:.7f} | '
            'lD: {lossD:.5f} | '
            'lL: {lossL:.5f} | '

        ).format(
            batch=i + 1,
            size=len(train_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            lossH=am_loss_hm.avg,
            lossD=am_loss_dm.avg,
            lossL=am_loss_lm.avg,

        )

        if DEBUG:
            if i == 1:
                break
        bar.next()
    bar.finish()

    loss_all["lossH"].append(am_loss_hm.avg)
    loss_all["lossD"].append(am_loss_dm.avg)
    loss_all["lossL"].append(am_loss_lm.avg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch Fine-tune: pruned 2D DetNet')
    # === FINETUNE-PRUNED CHANGE (b): pruned-model path ======================
    parser.add_argument(
        '--pruned_model_path',
        default=None,
        type=str,
        help='path to a pruned DetNet saved as a whole model object '
             '(torch.save(model, ...)); if set, fine-tune that 2D model'
    )
    # =======================================================================
    # Dataset setting
    parser.add_argument(
        '-dr',
        '--data_root',
        type=str,
        default="/home/chen/datasets/",
        help='dataset root directory'
    )
    parser.add_argument(
        "-trs",
        "--datasets_train",
        nargs="+",
        default=['cmu', 'rhd', 'gan'],
        type=str,
        help="sub datasets, should be listed in: [cmu|rhd|gan]"
    )

    parser.add_argument(
        "-tes",
        "--datasets_test",
        nargs="+",
        default=['rhd', 'stb', "do", "eo"],
        type=str,
        help="sub datasets, should be listed in: [rhd|stb|do|eo]"
    )

    # Miscs
    parser.add_argument(
        '-ckp',
        '--checkpoint',
        default='checkpoints',
        type=str,
        metavar='PATH',
        help='path to save checkpoint (default: checkpoint)'
    )

    parser.add_argument(
        '-sp',
        '--saved_prefix',
        default='ckp_detnet',
        type=str,
        metavar='PATH',
        help='path to save checkpoint (default: checkpoint)'
    )

    parser.add_argument(
        '-op',
        '--outpath',
        default='out_loss_auc',
        type=str,
        metavar='PATH',
        help='path to out_testset loss and auc (default: out_testset)'
    )

    parser.add_argument(
        '--snapshot',
        default=1, type=int,
        help='save models for every #snapshot epochs (default: 0)'
    )

    parser.add_argument(
        '-r', '--resume',
        dest='resume',
        action='store_true',
        help='whether to load checkpoint (default: none)'
    )
    parser.add_argument(
        '-e', '--evaluate',
        dest='evaluate',
        action='store_true',
        help='evaluate model on validation set'
    )

    # Training Parameters
    parser.add_argument(
        '-eid', '--evaluate_id',
        default=319,
        type=int,
        metavar='N',
        help='number of data loading workers (default: 8)'
    )
    parser.add_argument(
        '-c', '--clean',
        dest='clean',
        action='store_true',
        help='clean model on one gpu if trained on 2 gpus'
    )
    parser.add_argument(
        '-j', '--workers',
        default=8,
        type=int,
        metavar='N',
        help='number of data loading workers (default: 8)'
    )
    parser.add_argument(
        '--epochs',
        default=500,
        type=int,
        metavar='N',
        help='number of total epochs to run'
    )
    parser.add_argument(
        '-se', '--start_epoch',
        default=0,
        type=int,
        metavar='N',
        help='manual epoch number (useful on restarts)'
    )
    parser.add_argument(
        '-b', '--train_batch',
        default=32,
        type=int,
        metavar='N',
        help='train batchsize'
    )
    parser.add_argument(
        '-tb', '--test_batch',
        default=128,
        type=int,
        metavar='N',
        help='test batchsize'
    )

    parser.add_argument(
        '-lr', '--learning-rate',
        default=1e-3,
        type=float,
        metavar='LR',
        help='initial learning rate'
    )
    parser.add_argument(
        "--lr_decay_step",
        default=250,
        type=int,
        help="Epochs after which to decay learning rate",
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.1,
        help='LR is multiplied by gamma on schedule.'
    )

    parser.add_argument(
        '--det_loss',
        dest='det_loss',
        action='store_true',
        help='Calculate detnet loss',
        default=True
    )

    main(parser.parse_args())
