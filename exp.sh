#!/usr/bin/env bash

mkdir -p loss

declare -a MODEL=('TransE' 'RotatE' 'DistMult' 'ComplEx')
declare -a DATASET=('DDB14' 'WN18RR' 'FB15K237')

for n in 5 10 15 20
do
	for dataset in "${DATASET[@]}"
	do
		for model in "${MODEL[@]}"
		do
			file="main.py"
			data="Fed_data/${dataset}-Fed${n}.pkl"
			name="${dataset}_fed${n}_fed_${model}"
			cmd="python $file --data_path $data --name $name --run_mode FedR --model ${model} --num_client $n --early_stop_patience 5 --gpu 0"
			echo $cmd
			eval $cmd
		done
	done
done