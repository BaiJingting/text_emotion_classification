# !/bin/sh

# first step args
POS_START_DATE=`date -d -33day +%Y%m%d`
NEG_START_DATE=`date -d -46day +%Y%m%d`
NEU_START_DATE=`date -d -7day +%Y%m%d`
TRAIN_END_DATE=`date -d -6day +%Y%m%d`

# second step args
SECOND_STEP_END_DATE=`date -d -1day +%Y%m%d`
SECOND_STEP_START_DATE=`date -d -5day +%Y%m%d`

# train
nohup python train_first_step.py $POS_START_DATE $NEG_START_DATE $NEU_START_DATE $TRAIN_END_DATE &&
nohup python train_second_step.py $SECOND_STEP_START_DATE $SECOND_STEP_END_DATE

