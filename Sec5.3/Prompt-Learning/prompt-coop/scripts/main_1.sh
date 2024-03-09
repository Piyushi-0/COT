#!/bin/bash

cd ..

# custom config
DATA=../DATA
TRAINER=PLOT 

DATASET=$1
CFG=rn50  # config file
CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens
# SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)

for N in 4
do
for SHOTS in 1
do
for ktype in rbf
do
for khp in 10
do
for lda in 100
do
for SEED in 1 2 3
do
rm -r ./output/OP_N${N}/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
DIR=./output/OP_N${N}/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}

echo "Run this job and save the output to ${DIR}"
python ctrain.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
--lda ${lda} \
--ktype ${ktype} \
--khp ${khp} \
TRAINER.PLOT.N_CTX ${NCTX} \
TRAINER.PLOT.CSC ${CSC} \
TRAINER.PLOT.CLASS_TOKEN_POSITION ${CTP} \
DATASET.NUM_SHOTS ${SHOTS} \
TRAINER.PLOT.N ${N}
done
done
done
done
done
done
