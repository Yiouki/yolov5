nohup python3 /home/ruiz/codes/yolov5_perso/train.py \
	--epochs 1000 --weights '' --cfg models/custom_yolov5x.yaml \
	--data datasets/Invert_val/BaseTST/TRN/CV1/CV1_BaseTST.yaml \
	--project /data/hugo/Results/giga_invert_val/Refs/onlyReal \
	--name Ref_onlyReal_CV1_bs100_1000epochs_seed25_1_randomInit \
	--batch-size 100 --imgsz 512 --device 2 --seed 25 \
	--val-period 1 --early-stop --start-early-stop 100 --patience 350 \
	--method-fitness perso > nohups/20230407_1002_ref_gigaInvertVal_seed25_1_randomInit.out &

nohup python3 /home/ruiz/codes/yolov5_perso/train.py \
	--epochs 1000 --weights yolov5m.pt \
	--data datasets/Invert_val/BaseTST/TRN/CV1/CV1_BaseTST.yaml \
	--project /data/hugo/Results/giga_invert_val/Refs/onlyReal \
	--name Ref_onlyReal_CV1_bs100_1000epochs_seed25_1_sizeM \
	--batch-size 100 --imgsz 512 --device 3 --seed 25 \
	--val-period 1 --early-stop --start-early-stop 100 --patience 350 \
	--method-fitness perso > nohups/20230407_1002_ref_gigaInvertVal_seed25_1_sizeM.out &

nohup python3 /home/ruiz/codes/yolov5_perso/train.py \
	--epochs 1000 --weights yolov5m.pt \
	--data datasets/Invert_val/BaseTST/TRN/CV1/CV1_BaseTST.yaml \
	--project /data/hugo/Results/giga_invert_val/Refs/onlyReal \
	--name Ref_onlyReal_CV1_bs100_1000epochs_seed25_1_variableLR_10-2 \
	--batch-size 100 --imgsz 512 --device 1 --seed 25 \
	--val-period 1 --early-stop --start-early-stop 100 --patience 350 \
	--variable-lr --hyp /home/ruiz/codes/yolov5_perso/data/hyps/hr/hyp_hr_refVariableLR_10-2.yaml \
	--method-fitness perso > nohups/20230407_1526_ref_gigaInvertVal_seed25_1_variableLR_10-2.out &
