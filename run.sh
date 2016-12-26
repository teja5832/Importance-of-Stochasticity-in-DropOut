for h in abs abs_rev div div_rev drop drop_rev gauss
do
	for d in clusterincluster corners crescentfullmoon halfkernel outlier twospirals mnist
	do
		for a in ReLU Tanh Sigmoid
		do
	    	for p in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 
	    	do
				time th run.lua $h $d $a $p
	    	done
		done
   done
done