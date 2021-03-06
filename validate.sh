#!/bin/bash

#SBATCH --job-name="drd-validate"
#SBATCH --mail-type=end,fail
#SBATCH --mail-user="till.meyerzuwestram@artorg.unibe.ch"
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=01:00:0
##SBATCH --gres=gpu:gtx1080ti:1
##SBATCH --partition=gpu

## CUDA library
#module load CUDA/10.1.105-GCC-8.2.0-2.31.1
## optimized kernels for CUDA
#module load cuDNN/7.6.0.64-gcccuda-2019a
module load Python/3.7.2-GCCcore-8.2.0


declare -r BASEDIR=$HOME/drd
test -d $BASEDIR || exit 1
#declare -r WORKDIR=$(mktemp -d -p /data/users/$(whoami) "$(date +%Y_%m_%d-%H_%M_%S)-XXX")
declare -r WORKDIR=$1

test -d $WORKDIR || exit 1

echo -e "======================================================"
echo -e "WORKDIR:"
echo -e "\t$WORKDIR"
echo -e "======================================================"

(
	cd $BASEDIR
	pip install --user -r requirements.txt > /dev/null
	./run.py \
		--dir $BASEDIR \
		--state $WORKDIR/model.pth \
		--limit 1000 \
		--validate \
		--stats $WORKDIR/validate.csv
)

