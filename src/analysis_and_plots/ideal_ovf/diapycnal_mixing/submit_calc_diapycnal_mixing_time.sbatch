#!/bin/bash -l

#SBATCH --qos=normal                
#SBATCH --mem=150000                    
#SBATCH --ntasks=8                   
#SBATCH --output=dia_mix.out       
#SBATCH --time=360                    

export MPLBACKEND='Agg'
#python calc_diapycnal_mixing_time.py 
python calc_diapycnal_mixing_time_conc.py
unset MPLBACKEND

