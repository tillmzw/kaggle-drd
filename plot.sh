#!/bin/bash

#SBATCH --job-name="drd-plot"
#SBATCH --mail-type=end,fail
#SBATCH --mail-user="till.meyerzuwestram@artorg.unibe.ch"
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:0

module load Python/3.7.2-GCCcore-8.2.0


declare -r BASEDIR=$HOME/drd
test -d $BASEDIR || exit 1

declare -r WORKDIR=$1
test -d $WORKDIR || exit 1

echo -e "======================================================"
echo -e "WORKDIR:"
echo -e "\t$WORKDIR"
echo -e "======================================================"

(
	cd $BASEDIR
	pip install --user -r requirements.txt > /dev/null
	./plot_validation.py \
		-i $WORKDIR/validate.csv \
		-o $WORKDIR/validate_hist.png

)

echo "See attached file for validation information." | \
	/usr/bin/mail -s "Validation (Slurm $SLURM_JOB_NAME)" \
		-a $BASEDIR/$WORKDIR/validate_hist.png \
		till.meyerzuwestram@artorg.unibe.ch
