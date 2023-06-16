#!/bin/bash -l

#SBATCH --qos=normal                
#SBATCH --mem=150000                    
#SBATCH --ntasks=8                   
#SBATCH --output=dia_mix.out       
#SBATCH --time=360                    

python calc_osnap_IB_transport_T.py


