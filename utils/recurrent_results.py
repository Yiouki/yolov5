import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm

from utils.callbacks import Callbacks
from utils.general import LOGGER, Profile, colorstr, non_max_suppression, scale_boxes, xywh2xyxy
from utils.metrics import ConfusionMatrix, ap_per_class
from utils.plots import output_to_target, plot_images
from val import process_batch, save_one_json, save_one_txt

# TODO: capable d'avoir toutes les metrics possibles (F1, accuracy, recall, TP, FP...)
# TODO: ne pas oublier de sortir les positions des box dans un fichier npy
# TODO: Sauvegarder ça dans un CSV

class ResultsEachEpoch():
    def __init__(self,
                 data,
                 model,
                 dataloader,
                 single_cls=False,  # treat as single-class dataset
                 half=True,  # use FP16 half-precision inference
                 augment=False,  # augmented inference
                 batch_size=32,  # batch size
                 imgsz=512,  # inference size (pixels)
                 iou_thres=0.6,  # NMS IoU threshold
                 conf_thres=0.001,  # confidence threshold
                 compute_loss=None,
                 callbacks=Callbacks(),
                 task='val',  # train, val, test, speed or study
                 save_conf=False,  # save confidences in --save-txt labels
                 save_txt=False,  # save results to *.txt
                 save_hybrid=False,  # save label+prediction hybrid results to *.txt
                 save_dir=Path(''),
                 plots=True,
                 verbose=False,  # verbose output
                ) -> None:
        self.data = data
        self.model = model
        self.dataload = dataloader
        self.single_cls = single_cls
        self.half = half
        self.augment = augment
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.compute_loss = compute_loss
        self.callbacks = callbacks
        self.task = task
        self.save_conf = save_conf
        self.save_txt = save_txt
        self.save_hybrid = save_hybrid
        self.save_dir = f"{save_dir}/TST_MEGA_TEST"
        self.plots = plots
        self.verbose = verbose

        self.__config__()
        
    def __config__(self) -> None:
        self.device = next(self.model.parameters()).device  # get model device, PyTorch model
        self.pt = True
        self.jit = False
        self.engine = False
        
        self.half &= self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model.half() if self.half else self.model.float()

        # Configure
        self.model.eval()
        self.cuda = self.device.type != 'cpu'
        self.nc = 1 if self.single_cls else int(self.data['nc'])  # number of classes
        self.iouv = torch.linspace(0.5, 0.95, 10, device=self.device)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()

    def get(self, epoch):
        self.compute(epoch)
    
    def compute(self, epoch):
        seen = 0
        confusion_matrix = ConfusionMatrix(nc=self.nc)
        names = self.model.names if hasattr(self.model, 'names') else self.model.module.names  # get class names
        if isinstance(names, (list, tuple)):  # old format
            names = dict(enumerate(names))
        s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
        tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        dt = Profile(), Profile(), Profile()  # profiling times
        loss = torch.zeros(3, device=self.device)
        stats, ap, ap_class = [], [], []
        self.callbacks.run('on_val_start')
        pbar = tqdm(self.dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
            self.callbacks.run('on_val_batch_start')
            with dt[0]:
                if self.cuda:
                    im = im.to(self.device, non_blocking=True)
                    targets = targets.to(self.device)
                im = im.half() if self.half else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                nb, _, height, width = im.shape  # batch size, channels, height, width

            # Inference
            with dt[1]:
                preds, train_out = self.model(im) if self.compute_loss else (self.model(im, augment=self.augment), None)

            # Loss
            if self.compute_loss:
                loss += self.compute_loss(train_out, targets)[1]  # box, obj, cls

            # NMS
            targets[:, 2:] *= torch.tensor((width, height, width, height), device=self.device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if self.save_hybrid else []  # for autolabelling
            with dt[2]:
                preds = non_max_suppression(preds,
                                            self.conf_thres,
                                            self.iou_thres,
                                            labels=lb,
                                            multi_label=True,
                                            agnostic=self.single_cls,
                                            max_det=self.max_det)

            # Metrics
            for si, pred in enumerate(preds):
                labels = targets[targets[:, 0] == si, 1:]
                nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
                path, shape = Path(paths[si]), shapes[si][0]
                correct = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init
                seen += 1

                if npr == 0:
                    if nl:
                        stats.append((correct, *torch.zeros((2, 0), device=self.device), labels[:, 0]))
                        if self.plots:
                            confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                    continue

                # Predictions
                if self.single_cls:
                    pred[:, 5] = 0
                predn = pred.clone()
                scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

                # Evaluate
                if nl:
                    tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                    scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                    correct = process_batch(predn, labelsn, self.iouv)
                    if self.plots:
                        confusion_matrix.process_batch(predn, labelsn)
                stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

                # Save/log
                if self.save_txt:
                    save_one_txt(predn, self.save_conf, shape, file=self.save_dir / 'labels' / f'{path.stem}.txt')
                self.callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

            # Plot images
            if self.plots and batch_i < 3:
                plot_images(im, targets, paths, self.save_dir / f'val_batch{batch_i}_labels.jpg', names)  # labels
                plot_images(im, output_to_target(preds), paths, self.save_dir / f'val_batch{batch_i}_pred.jpg', names)  # pred

            self.callbacks.run('on_val_batch_end', batch_i, im, targets, paths, shapes, preds)

        # Compute metrics
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=self.plots, save_dir=self.save_dir, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(int), minlength=self.nc)  # number of targets per class

        if self.verbose:
            self.print_results(confusion_matrix, stats, names, seen, nt, ap_class,
                               tp, fp, f1,
                               p, r, ap50, ap,
                               mp, mr, map50, map)

        # Return results
        self.model.float()  # for training
        s = f"\n{len(list(self.save_dir.glob('labels/*.txt')))} labels saved to {self.save_dir / 'labels'}" if self.save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        maps = np.zeros(self.nc) + map
        for i, c in enumerate(ap_class):
            maps[c] = ap[i]
        return (mp, mr, map50, map, *(loss.cpu() / len(self.dataloader)).tolist()), maps

    def print_results(self, confusion_matrix, stats, names, seen, nt, ap_class, dt,
                      tp, fp, f1,
                      p, r, ap50, ap,
                      mp, mr, map50, map):
        # Print results
        pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
        LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
        if nt.sum() == 0:
            LOGGER.warning(f'WARNING ⚠️ no labels found in {self.task} set, can not compute metrics without labels')

        # Print results per class
        if self.verbose and self.nc > 1 and len(stats):
            for i, c in enumerate(ap_class):
                LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

        # Print speeds
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        shape = (self.batch_size, 3, self.imgsz, self.imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

        # Plots
        if self.plots:
            confusion_matrix.plot(save_dir=self.save_dir, names=list(names.values()))
            self.callbacks.run('on_val_end', nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)