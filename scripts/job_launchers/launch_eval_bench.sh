#!/bin/bash

# This script launches evaluation Slurm jobs for merged and/or mixed models across various benchmarks and experimental axes.
#
# The user can specify:
#   - model subset (experts, 2domains, 3domains, 4domains)
#   - the set of benchmarks to evaluate on
#   - base models
#   - fine-tuning strategies
#   - number of training steps (defining the training budget, budget=steps*128)

subset=4domains # experts, 2domains, 3domains, 4domains

BENCHMARKS=(
    gqa
    # vqav2_val_lite
    # vizwiz_vqa_val
    # ok_vqa_val2014
    # textvqa_val
    # ocrbench
    # docvqa_val
    # infovqa_val
    # cv_bench_2d
    # pope
    # chartqa
    # mme
    # vmcbench
    # mmstar
)
MODELS=(
    qwen2_2b
    #intern35_2b
    #qwen2_7b
    #intern35_8b
)
SFT_STRATEGIES=(
    lora
    #full
)
STEPS_LIST=(
    #80
    #400
    800
)
eval_merged=1 # Whether to evaluate merged-proxy models
eval_mixed=1 # Whether to evaluate mixed models

timelimit="02:00:00"
# For merged models
METHOD=task_arithmetic
MERGED_TYPE=merged
# ----------------------------------------------------------------------------------------------- #

BENCHMARKS_CSV=$(IFS=, ; echo "${BENCHMARKS[*]}")
N=${#BENCHMARKS[@]}

# Ask user for confirmation
printf "Launching eval jobs for: \nSubset: ${subset} \nModels: ${MODELS[*]} \nSFT strategies: ${SFT_STRATEGIES[*]} \nSteps: ${STEPS_LIST[*]} \nBenchmarks: ${BENCHMARKS_CSV} \nTimeLimit: ${timelimit} \n"
read -p "Are you sure? (enter): " confirmation
if [[ $confirmation != "" ]]; then
    exit 1
fi

submit_job () {
    
    # Submit job to Slurm and log the job ID and command
    # jobid=$(sbatch --parsable "$@")
    # echo "$jobid : $*" >> $LOGDIR/jobids.log
    # echo "$jobid : $*" >> logs/joblogs/jobids_$(date +%Y-%m-%d).log
    # echo $jobid

    # Dry run
    # echo "[DRY RUN] sbatch args: $*"

    # Run script directly (for debugging). Ignore the Slurm args and just run the script with its arguments (ie. ignore everything before "scripts/" and keep everything after)
    # Extract the part of the command after "scripts/" and run it "bash ..."
    cmd=$(echo "$@" | sed -n 's/.*\(scripts\/.*\).*/\1/p')
    eval "bash $cmd"
}

if [ "$subset" == "experts" ]; then

    DOMAINS=(
        "chart"
        "counting"
        "general"
        "ocr"
    )
    for BASE_MODEL in "${MODELS[@]}"; do
        for DOMAIN in "${DOMAINS[@]}"; do
            for SFT_STRATEGY in "${SFT_STRATEGIES[@]}"; do
                for STEPS in "${STEPS_LIST[@]}"; do

                    N_SAMPLES=$((STEPS * 128))
                    BUDGET_ID=$((STEPS / 8))k # 10k, 50k, 100k
                    EXP_NAME=exp_${BASE_MODEL}_${BUDGET_ID}

                    LOGDIR=logs/${EXP_NAME}

                    # Job-array over benchmarks
                    submit_job --array=0-$(($N - 1)) \
                            -t ${timelimit} \
                            --job-name=eval_${BASE_MODEL}_${SFT_STRATEGY}_${DOMAIN} \
                            --output=$LOGDIR/eval_bench_expert/%A_%x/%A_%a_%x.out --error=$LOGDIR/eval_bench_expert/%A_%x/%A_%a_%x.err \
                            scripts/experts/eval_bench_expert.sh ${BASE_MODEL} ${DOMAIN} ${BENCHMARKS_CSV} ${SFT_STRATEGY} ${STEPS}

                done
            done
        done
    done


elif [ "$subset" == "2domains" ]; then
    
    DOMAIN_COUPLES=(
        "general,ocr"
    )
    for BASE_MODEL in "${MODELS[@]}"; do
        for COUPLE in "${DOMAIN_COUPLES[@]}"; do
            for SFT_STRATEGY in "${SFT_STRATEGIES[@]}"; do
                for STEPS in "${STEPS_LIST[@]}"; do

                    N_SAMPLES=$((STEPS * 128))
                    BUDGET_ID=$((STEPS / 8))k # 10k, 50k, 100k
                    EXP_NAME=exp_${BASE_MODEL}_${BUDGET_ID}

                    LOGDIR=logs/${EXP_NAME}

                    DOM1=$(echo $COUPLE | cut -d',' -f1)
                    DOM2=$(echo $COUPLE | cut -d',' -f2)

                    for BENCHMARK in "${BENCHMARKS[@]}"; do

                        # Mixed
                        if [ "$eval_mixed" -eq 1 ]; then
                        submit_job --job-name=eval_bench_mixed2_${DOM1}_${DOM2} \
                            -t ${timelimit} \
                            --output=$LOGDIR/eval_bench_mixed2/%A_%x/%A_%a_%x.out --error=$LOGDIR/eval_bench_mixed2/%A_%x/%A_%a_%x.err \
                            scripts/mixed2/eval_bench_mixed2.sh ${BASE_MODEL} ${DOM1} ${DOM2} ${BENCHMARK} ${SFT_STRATEGY} ${STEPS} ${EXP_NAME}
                        fi

                        # Merged
                        if [ "$eval_merged" -eq 1 ]; then
                        submit_job --job-name=eval_bench_merged2_${DOM1}_${DOM2} \
                            -t ${timelimit} \
                            --output=$LOGDIR/eval_bench_merged2/%A_%x/%A_%a_%x.out --error=$LOGDIR/eval_bench_merged2/%A_%x/%A_%a_%x.err \
                            scripts/merged2/eval_bench_merged2.sh ${BASE_MODEL} ${DOM1} ${DOM2} ${BENCHMARK} ${SFT_STRATEGY} ${STEPS} ${METHOD} ${MERGED_TYPE} ${EXP_NAME}
                        fi
                    done
                    
                done
                
            done
        done
    done

elif [ "$subset" == "3domains" ]; then
    DOMAIN_TRIPLETS=(
        "counting,general,ocr"
    )
    for BASE_MODEL in "${MODELS[@]}"; do
        for TRIPLET in "${DOMAIN_TRIPLETS[@]}"; do
            for SFT_STRATEGY in "${SFT_STRATEGIES[@]}"; do
                for STEPS in "${STEPS_LIST[@]}"; do

                    N_SAMPLES=$((STEPS * 128))
                    BUDGET_ID=$((STEPS / 8))k # 10k, 50k, 100k
                    EXP_NAME=exp_${BASE_MODEL}_${BUDGET_ID}

                    LOGDIR=logs/${EXP_NAME}

                    DOM1=$(echo $TRIPLET | cut -d',' -f1)
                    DOM2=$(echo $TRIPLET | cut -d',' -f2)
                    DOM3=$(echo $TRIPLET | cut -d',' -f3)

                    for BENCHMARK in "${BENCHMARKS[@]}"; do

                        # Mixed
                        if [ "$eval_mixed" -eq 1 ]; then
                        submit_job --job-name=eval_bench_mixed3_${DOM1}_${DOM2}_${DOM3}  \
                            -t ${timelimit} \
                            --output=$LOGDIR/eval_bench_mixed3/%A_%x/%A_%a_%x.out --error=$LOGDIR/eval_bench_mixed3/%A_%x/%A_%a_%x.err \
                            scripts/mixed3/eval_bench_mixed3.sh ${BASE_MODEL} ${DOM1} ${DOM2} ${DOM3} ${BENCHMARK} ${SFT_STRATEGY} ${STEPS} ${EXP_NAME}
                        fi

                        # Merged
                        if [ "$eval_merged" -eq 1 ]; then
                        LOG=$LOGDIR/eval_bench_merged3/%A_%x/%A_%a_%x
                        submit_job --job-name=eval_bench_merged3_${DOM1}_${DOM2}_${DOM3} \
                            -t ${timelimit} \
                            --output=${LOG}.out --error=${LOG}.err \
                            scripts/merged3/eval_bench_merged3.sh ${BASE_MODEL} ${DOM1} ${DOM2} ${DOM3} ${BENCHMARK} ${SFT_STRATEGY} ${STEPS} ${METHOD} ${MERGED_TYPE} ${EXP_NAME}
                        fi

                    done
                    
                done
                
            done
        done
    done

elif [ "$subset" == "4domains" ]; then
    DOMAIN_LIST=(
        "chart,counting,general,ocr"
    )
    for BASE_MODEL in "${MODELS[@]}"; do
        for QUADRUPLET in "${DOMAIN_LIST[@]}"; do
            for SFT_STRATEGY in "${SFT_STRATEGIES[@]}"; do
                for STEPS in "${STEPS_LIST[@]}"; do

                    N_SAMPLES=$((STEPS * 128))
                    BUDGET_ID=$((STEPS / 8))k # 10k, 50k, 100k
                    EXP_NAME=exp_${BASE_MODEL}_${BUDGET_ID}

                    LOGDIR=logs/${EXP_NAME}

                    DOM1=$(echo $QUADRUPLET | cut -d',' -f1)
                    DOM2=$(echo $QUADRUPLET | cut -d',' -f2)
                    DOM3=$(echo $QUADRUPLET | cut -d',' -f3)
                    DOM4=$(echo $QUADRUPLET | cut -d',' -f4)

                    for BENCHMARK in "${BENCHMARKS[@]}"; do

                        # Mixed
                        if [ "$eval_mixed" -eq 1 ]; then
                        submit_job --job-name=eval_bench_mixed4_${DOM1}_${DOM2}_${DOM3}_${DOM4}  \
                            -t ${timelimit} \
                            --output=$LOGDIR/eval_bench_mixed4/%A_%x/%A_%a_%x.out --error=$LOGDIR/eval_bench_mixed4/%A_%x/%A_%a_%x.err \
                            scripts/mixed4/eval_bench_mixed4.sh ${BASE_MODEL} ${DOM1} ${DOM2} ${DOM3} ${DOM4} ${BENCHMARK} ${SFT_STRATEGY} ${STEPS} ${EXP_NAME}
                        fi

                        # Merged
                        if [ "$eval_merged" -eq 1 ]; then
                        LOG=$LOGDIR/eval_bench_merged4/%A_%x/%A_%a_%x
                        submit_job --job-name=eval_bench_merged4_${DOM1}_${DOM2}_${DOM3}_${DOM4} \
                            -t ${timelimit} \
                            --output=${LOG}.out --error=${LOG}.err \
                            scripts/merged4/eval_bench_merged4.sh ${BASE_MODEL} ${DOM1} ${DOM2} ${DOM3} ${DOM4} ${BENCHMARK} ${SFT_STRATEGY} ${STEPS} ${METHOD} ${MERGED_TYPE} ${EXP_NAME}
                        fi

                    done
                    
                done
                
            done
        done
    done
else
    echo "Unknown subset: ${subset}"
    exit 1
fi