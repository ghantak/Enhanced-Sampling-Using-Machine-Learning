Steps to follow the Metadynamics Simulations of Protein Using Machine Learning ( Gaussian Process Regression) in NAMD

First Run WT metadynamics for about 30% of the estimated simulation time 

Extract the CVs and bias in a file using the python code current_cv_bias_extract.py

python current_cv_bias_extract.py

This will generate the update_bias.dat file 

Train the CVs and bias values using train_gpr_gpu.py using slurm script

sbatch train_gpr.slurm

This will generate a .pth file as a training output file and 

Now, we need to modify the NAMD configuration file to add the gpr_control.tcl file (check the file System_NPT_MetaD2.conf)



