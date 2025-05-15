#####Arguments#####
ID=1ssc
REC_MSA=../../../data/design/1ssc/1ssc_receptor.a3m
REC_FASTA=../../../data/design/1ssc/1ssc_receptor.fasta
MAX_RECYCLES=3 #max_recycles (default=3)
BINDER_LENGTH=11
NITER=1000
BATCH_SIZE=2
#PARAMS=/home/bryant/Desktop/rare_fold/params/params20000.npy
PARAMS=/proj/berzelius-2023-267/users/x_patbr/results/rare_fold/train/rare_sampling/params20000.npy
RARE_AAS="MSE,MLY,PTR,SEP,TPO,MLZ,ALY,HIC,HYP,M3L,PFF,MHO"
CYCLIC=True
OUTDIR=../../../data/design/1ssc/


#######First make MSA features#######
python3 ../input/make_msa_seq_feats.py --input_fasta_path $REC_FASTA \
--input_msas $REC_MSA --outdir $OUTDIR

MSA_FEATS=$OUTDIR/msa_features.pkl

#######Then design#######
python3 ./mc_design_improved.py --predict_id $ID \
--MSA_feats $MSA_FEATS \
--num_recycles $MAX_RECYCLES \
--binder_length $BINDER_LENGTH \
--num_iterations $NITER \
--batch_size $BATCH_SIZE \
--params $PARAMS \
--rare_AAs $RARE_AAS \
--cyclic_offset $CYCLIC \
--outdir $OUTDIR

