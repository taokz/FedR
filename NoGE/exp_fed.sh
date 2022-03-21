#!/usr/bin/env bash

mkdir -p loss

#declare -a DATA=('DDB14' 'FB15K237' 'WN18RR')
declare -a DATA=('DDB14')

for n in 3 5 10 15 20
do
	for data in "${DATA[@]}"
		do
			file="main.py"
			data_path="../Fed_data/${data}-Fed${n}.pkl"
			name="${data}_fed${n}_fed_NoGE"
			cmd="python $file --data_path $data_path --name $name --run_mode FedNoGE --num_client $n --early_stop_patience 5 --gpu 0"
			echo $cmd
			eval $cmd
		done
done