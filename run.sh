
PATH_DATA='/home/xxx/FFpp/SegDataV3'
PATH_CFG_FILE="seg/cfgs/ocrnet/cfgs_df_xception.py"


#####################################################################
## setting
jpegCW=0.5
GBlurW=0.5
jpegDW=0.5
GNoiseW=0
colorJW=0

resize=299	#pretrain resize: 299
cropSize=299
trainBSize=32

model='LiSiam_A'

#####################################################################
## step 1

clsWeight=1.0
clsWeight2=0.0
invWeight=0
auxWeight=0
auxWeight2=0
segWeight=0
segWeight2=0


max_epochs=90
PATH_BACKUP="checkpoints/C40XAS_A"
CHECKPOINTSPATH="checkpoints/xPre/C40XAll_AS5_sgd27_v1/model_end.pth"
python -u seg/trainG9.3.py --cfgfilepath $PATH_CFG_FILE --checkpointspath $CHECKPOINTSPATH --net $model \
			       --PATH_data $PATH_DATA  --backupdir $PATH_BACKUP --trainFile 'trainUPDB' \
			       --trainBSize=$trainBSize --testBSize=100 --cropSize $cropSize --resize $resize --scaleRangeL 1 --scaleUpper 0.8  --scaleLower 1.2 \
				   --typeOPT 'sgd' --LR 0.002 --adjustLR 1 --typeLR 'poly' --periodLR 'iteration' \
				   --clsWeight=$clsWeight --clsWeight2=$clsWeight2 --invWeight=$invWeight --auxWeight=$auxWeight --auxWeight2=$auxWeight2 --segWeight=$segWeight --segWeight2=$segWeight2 \
				   --jpegCW=$jpegCW --GBlurW=$GBlurW --jpegDW=$jpegDW  --GNoiseW=$GNoiseW  --colorJW=$colorJW  \
				   --labelFolder 'labelsGTC40a' --trainFolder 'FFppDFaceC40a' --trainLabelFile 'C40a.dat' --Dataset 'C40' \
			       --loginterval=1000 --max_epochs=$max_epochs --FROZEN 2 --evalLabel 0  --NUM_gpus 2 --tqdm True

echo $(date)


#####################################################################
## step 2

clsWeight=1.0
clsWeight2=0.0
invWeight=1.0
auxWeight=0.4
auxWeight2=0.0
segWeight=1.0
segWeight2=0.0


max_epochs=1800
PATH_BACKUP="checkpoints/C40XAS_A_1117s3"
CHECKPOINTSPATH="checkpoints/C40XAS_A/model_end.pth"

python -u seg/trainG9.3.py --cfgfilepath $PATH_CFG_FILE --checkpointspath $CHECKPOINTSPATH --net $model \
			       --PATH_data $PATH_DATA  --backupdir $PATH_BACKUP --trainFile 'trainUPDB' \
			       --trainBSize=$trainBSize --testBSize=100 --cropSize $cropSize --resize $resize --scaleRangeL 1 --scaleUpper 0.8  --scaleLower 1.2 \
				   --typeOPT 'sgd' --LR 0.002 --adjustLR 1 --typeLR 'poly' --periodLR 'iteration' \
				   --clsWeight=$clsWeight --clsWeight2=$clsWeight2 --invWeight=$invWeight --auxWeight=$auxWeight --auxWeight2=$auxWeight2 --segWeight=$segWeight --segWeight2=$segWeight2 \
				   --jpegCW=$jpegCW --GBlurW=$GBlurW --jpegDW=$jpegDW  --GNoiseW=$GNoiseW  --colorJW=$colorJW  \
				   --labelFolder 'labelsGTC40a' --trainFolder 'FFppDFaceC40a' --trainLabelFile 'C40a.dat' --Dataset 'C40'   \
			       --loginterval=1000 --max_epochs=$max_epochs --FROZEN 0 --evalLabel 1  --NUM_gpus 2

echo $(date)
