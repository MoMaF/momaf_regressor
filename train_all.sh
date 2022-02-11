for lr in 5e-5 #1e-6 5e-6 1e-5 5e-5
do
    for field in content-noyearnopers # content-orig content-noyear 
    do
	mname=momaf_${lr}_${field}
	echo "echo starting $mname"
	echo "python3 train.py --cheat --lr $lr --steps 1500 --field $field --save-to models/${mname}.cheat.model > models/${mname}.cheat.out 2> models/${mname}.cheat.err"
	### echo "python3 train.py --lr $lr --steps 1500 --field $field --sep --save-to models/${mname}_sep.model > models/${mname}_sep.out 2> models/${mname}_sep.err"
    done
done
