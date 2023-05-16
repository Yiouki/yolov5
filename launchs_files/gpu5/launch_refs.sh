nohup python3 /home/ruiz/codes/yolov5_perso/train.py \
	--epochs 350 --weights yolov5x.pt \
	--data datasets/Invert_val/BaseTST/TRN/CV1/CV1_BaseTST.yaml \
	--project /data/hugo/Results/giga_invert_val/Refs/onlyReal \
	--name Ref_onlyReal_CV1_bs100_350epochs_seed25_1 \
	--batch-size 100 --imgsz 512 --device 2 --seed 25 \
	--val-period 1 --early-stop --start-early-stop 100 --patience 350 \
	--method-fitness perso > nohups/20230412_1848_ref_gigaInvertVal_seed25_1.out &

nohup python3 /home/ruiz/codes/yolov5_perso/train.py \
	--epochs 350 --weights yolov5x.pt \
	--data datasets/Invert_val/BaseTST/TRN/CV1/CV1_BaseTST.yaml \
	--project /data/hugo/Results/giga_invert_val/Refs/onlyReal \
	--name Ref_onlyReal_CV1_bs100_350epochs_seed10_2 \
	--batch-size 100 --imgsz 512 --device 3 --seed 10 \
	--val-period 1 --early-stop --start-early-stop 100 --patience 350 \
	--method-fitness perso > nohups/20230412_1848_ref_gigaInvertVal_seed10_2.out &

nohup python3 /home/ruiz/codes/yolov5_perso/train.py \
	--epochs 350 --weights yolov5x.pt \
	--data datasets/Invert_val/BaseTST/TRN/CV1/CV1_BaseTST.yaml \
	--project /data/hugo/Results/giga_invert_val/Refs/onlyReal \
	--name Ref_onlyReal_CV1_bs100_350epochs_seed95_3 \
	--batch-size 100 --imgsz 512 --device 4 --seed 95 \
	--val-period 1 --early-stop --start-early-stop 100 --patience 350 \
	--method-fitness perso > nohups/20230412_1848_ref_gigaInvertVal_seed95_3.out &