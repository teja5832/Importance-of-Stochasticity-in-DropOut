This folder contains the following files:

	1.	README.txt - This file.
	2.	run.lua - The script that runs the code - trains the NN, and produces the result.

		Usage: th run.lua <heuristic> <dataset> <activation> <p>
		

		It produces folder with the name 
			<date>_<heuristic>_<dataset>_<activation>_p<p>
		This folder contains the following files:
		- eplts.png: Error Plots
		- tr_eplt.png: Training Error Plot
		- tst_eplt.png: Test Error Plot
		- lplts.png: Loss Plots
		- tr_lplt.png: Training Loss Plot
		- tst_lplt.png: Test Loss Plot
		- mlp.dat: Serialized Model. Can be loaded with model = torch.load('mlp.dat') in torch7.
		- test_losses.dat: Serialized Test Loss.
		- train_losses.dat: Serialized Training Loss
        - test_error.dat: Serialized Test Error
        - train_error.dat: Serialized Training Error
	3.	The seven heuristics
		- drop.lua : Drop weights in the band  [µ-pσ, µ+pσ] during training. Halve the weights while testing.
		- drop_rev.lua : Drop weights outside the band  [µ-pσ, µ+pσ] during training. Halve the weights while testing.
		- div.lua : Drop weights in the band  [µ-pσ, µ+pσ] during training. Divide weights by (#of times dropped) + 1 while testing.
		- div_rev.lua : Drop weights outside the band  [µ-pσ, µ+pσ] during training. Divide weights by  (#of times dropped) + 1 while testing.
		- abs.lua : Drop weights whose absolute values are in the band  [µ-pσ, µ+pσ] during training.
		- abs_rev.lua : Drop weights whose absolute values are outside the band  [µ-pσ, µ+pσ] during training. 
		- gauss.lua : Use a Gaussian to sample px(#of weights) numbers, and drop the weights in a small band of each sampled number during training.
	4. run.sh - Runs all possible test cases for run.lua - by a four fold for loop. BEWARE, THIS CAN GENERATE AN OBSCENE AMOUNT OF DATA!
	5. make_dataset : This directory contains all the MATLAB® scripts to generate the random data, namely:
		- clusterincluster.m
		- corners.m
		- crescentfullmoon.m
		- csvwrite_with_headers.m
		- halfkernel.m
		- outlier.m
		- twospirals.m
	   It also contains the script to generate and plot all of them, namely, make_data.m
	   It also has the helper script csvwrite_with_headers.m I have also zipped up the datasets in csv; however, if you want to generate them, use
	   		cd make_dataset
	   		matlab -nodesktop -nosplash -r "run make_data.m; quit;" 
	 6. mnist.t7 : Contains MNIST Dataset. Do not touch!

References for the code:
* 6 functions for generating artificial datasets, by Jeroen Kools, 23 Apr 2013, retrieved from 
http://www.mathworks.com/matlabcentral/fileexchange/41459-6-functions-for-generating-artificial-datasets

* CSV with column headers, by Keith Brady, 06 Jan 2011, retrieved from 
http://in.mathworks.com/matlabcentral/fileexchange/29933-csv-with-column-headers

* Torch7 implementation of DropConnect by John-Alexander M. Assael, retrieved from https://github.com/iassael/torch-dropconnect

*  MNIST Data download code from https://github.com/oxford-cs-ml-2015/practical3
