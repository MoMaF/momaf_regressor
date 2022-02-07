for lr in 1e-6 5e-6 1e-5 5e-5
do
    for field in content-orig content-noyear content-noyearnopers
    do
	mname=momaf_${lr}_${field}
	echo "echo starting $mname"
	echo "python3 train.py --lr $lr --steps 1500 --field $field --save-to models/${mname}.model > models/${mname}.out 2> models/${mname}.err"
	echo "python3 train.py --lr $lr --steps 1500 --field $field --sep --save-to models/${mname}_sep.model > models/${mname}_sep.out 2> models/${mname}_sep.err"
    done
done
