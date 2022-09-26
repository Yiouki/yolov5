from pathlib import Path
import torch
from tqdm import tqdm

from utils.callbacks import Callbacks
from utils.general import Profile, non_max_suppression, scale_boxes, xywh2xyxy
from utils.metrics import ConfusionMatrix
from utils.plots import output_to_target, plot_images
from val import process_batch, save_one_json, save_one_txt

# TODO: capable d'avoir toutes les metrics possibles (F1, accuracy, recall, TP, FP...)
# TODO: Sauvegarder ça dans un CSV

class ResultsEachEpoch():
    def __init__(self,
                 epoch,
                 data,
                 model,
                 dataloader,
                 half=True,  # use FP16 half-precision inference
                 single_cls=False,  # treat as single-class dataset
                 callbacks=Callbacks(),
                 compute_loss=None,
                 augment=False,  # augmented inference
                 save_conf=False,  # save confidences in --save-txt labels
                 save_txt=False,  # save results to *.txt
                 save_json=False,  # save a COCO-JSON results file
                 save_hybrid=False,  # save label+prediction hybrid results to *.txt
                 save_dir=Path(''),
                 plots=True,
                ) -> None:
        self.epoch = epoch
        self.data = data
        self.model = model
        self.half = half
        self.pt = True
        self.jit = False
        self.engine = False
        self.device = next(model.parameters()).device  # get model device, PyTorch model
        self.single_cls = single_cls
        self.dataload = dataloader
        self.callbacks = callbacks
        self.compute_loss = compute_loss
        self.augment = augment
        self.save_conf = save_conf
        self.save_txt = save_txt
        self.save_json = save_json
        self.save_hybrid = save_hybrid
        self.save_dir = save_dir
        self.plots = plots

        self.__config__()
        
    def __config__(self) -> None:
        self.half &= self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model.half() if self.half else self.model.float()

        # Configure
        self.model.eval()
        self.cuda = self.device.type != 'cpu'
        self.nc = 1 if self.single_cls else int(self.data['nc'])  # number of classes
        self.iouv = torch.linspace(0.5, 0.95, 10, device=self.device)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()

    def compute(self):
        seen = 0
        confusion_matrix = ConfusionMatrix(nc=self.nc)
        names = self.model.names if hasattr(self.model, 'names') else self.model.module.names  # get class names
        if isinstance(names, (list, tuple)):  # old format
            names = dict(enumerate(names))
        class_map = list(range(1000))
        s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
        tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        dt = Profile(), Profile(), Profile()  # profiling times
        loss = torch.zeros(3, device=self.device)
        jdict, stats, ap, ap_class = [], [], [], []
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
                if self.save_json:
                    save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
                self.callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

            # Plot images
            if self.plots and batch_i < 3:
                plot_images(im, targets, paths, self.save_dir / f'val_batch{batch_i}_labels.jpg', names)  # labels
                plot_images(im, output_to_target(preds), paths, self.save_dir / f'val_batch{batch_i}_pred.jpg', names)  # pred

            self.callbacks.run('on_val_batch_end', batch_i, im, targets, paths, shapes, preds)