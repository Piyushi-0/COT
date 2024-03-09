for lda in 0.1
do
	for khp in 0.1
	do
		for lr in 0.005
		do
			for n_epochs in 20
			do
				echo "khp" $khp "lda" $lda "lr" $lr "n_epochs" $n_epochs 
				python3 main.py --khp $khp --lda $lda --lr $lr --n_epochs $n_epochs
			done
		done
	done
done
