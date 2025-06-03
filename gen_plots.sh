#!/bin/bash

set -euo pipefail

datasets=("MINI" "SMALL" "MEDIUM" "LARGE" "EXTRALARGE")

for dataset in "${datasets[@]}"; do
	echo "Generating plots for dataset: $dataset"
	python3 gen_plots.py --output "plots/$dataset.pdf" \
		--show-sdev \
		--no-validation-legend \
		--no-boostP2P \
		--datasets "$dataset" \
		-- data/compare?.csv
done

for dataset in "${datasets[@]}"; do
	echo "Generating plots for dataset: $dataset with BoostP2P"
	python3 gen_plots.py --output "plots/${dataset}_with_boostP2P.pdf" \
		--show-sdev \
		--no-validation-legend \
		--datasets "$dataset" \
		-- data/compare?.csv
done

echo "Generating a combined plot for MINI and EXTRALARGE datasets (presented in the paper)"
python3 gen_plots.py --output "plots/combined_MINI_EXTRALARGE.pdf" \
	--show-sdev \
	--no-validation-legend \
	--mark-noarr \
	--datasets "MINI" "EXTRALARGE" \
	--no-boostP2P \
	-- data/compare?.csv
