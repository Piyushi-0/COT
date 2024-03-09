device=$1
seed_set_idx=$2
if [ $seed_set_idx = 0 ]; then
    	seed_set=(10 11 12)
elif [ $seed_set_idx = 1 ]; then
    	seed_set=(20 21 22 )
elif [ $seed_set_idx = 2 ]; then
    	seed_set=(30 31 32 )
else
	seed_set=(40 41 42)
fi

for seed in ${seed_set[@]};
do
    for nz in 50 100 200 400 600
    do
        python correctness_exp_v3-tabak.py --lr 1e-3 -rx 1e3 -nz $nz -s $seed -d $device -bs 50
    done
done
