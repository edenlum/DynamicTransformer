##!/bin/bash
#
## Define block sizes and iterations to try
#block_sizes=(2 3)
#iters=(1 2 3)
#
## Learning rate found to be optimal
#learning_rate=1e-3
#
## Function to get a list of available GPU indices
#get_available_gpus() {
#    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '{if ($1 == 0) print NR-1}'
#}
#
## Get the list of available GPUs
#available_gpus=($(get_available_gpus))
#num_available_gpus=${#available_gpus[@]}
#
#if [ $num_available_gpus -eq 0 ]; then
#    echo "No available GPUs found."
#    exit 1
#fi
#
## Loop through the block sizes and iterations and run a training job for each
#count=0
#for block_size in "${block_sizes[@]}"; do
#    for iter in "${iters[@]}"; do
#        if [ $count -ge $num_available_gpus ]; then
#            break 2
#        fi
#
#        # Select an available GPU (cycle through available GPUs)
#        gpu=${available_gpus[$count]}
#
#        # Run the training script with the current block size, iteration, learning rate, and selected GPU
#        echo "Running on GPU $gpu with block_size=$block_size and iter=$iter"
#        CUDA_VISIBLE_DEVICES=$gpu python train_router.py model.block_size=$block_size model.iters=$iter model.learning_rate=$learning_rate &
#
#        count=$((count + 1))
#    done
#done
#
## Wait for all background jobs to complete
#wait
#
#echo "All training jobs are complete."

#!/bin/bash

# Define block sizes and iterations to try
swap_layers=(0 4 8 12)

# Learning rate found to be optimal
learning_rate=1e-3

# Function to get a list of available GPU indices
get_available_gpus() {
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '{if ($1 == 0) print NR-1}'
}

# Get the list of available GPUs
available_gpus=($(get_available_gpus))
num_available_gpus=${#available_gpus[@]}

if [ $num_available_gpus -eq 0 ]; then
    echo "No available GPUs found."
    exit 1
fi

# Loop through the block sizes and swap layers and run an evaluation job for each
count=0
for swap_layer in "${swap_layers[@]}"; do
    if [ $count -ge $num_available_gpus ]; then
        break 2
    fi

    # Select an available GPU (cycle through available GPUs)
    gpu=${available_gpus[$count]}

    # Run the evaluation script with the current block size, swap layer, learning rate, and selected GPU
    echo "Running on GPU $gpu with block_size=$block_size and swap_layer=$swap_layer"
    CUDA_VISIBLE_DEVICES=$gpu python skip_layers_eval.py model.swap_layer=$swap_layer &

    count=$((count + 1))
done

# Wait for all background jobs to complete
wait

echo "All evaluation jobs are complete."
