#!/bin/bash

# This script launches training Slurm jobs.
#
# The user can specify:
#   - model subset (experts, 2domains, 3domains, 4domains)
#   - the set of benchmarks to evaluate on
#   - base models
#   - fine-tuning strategies
#   - number of training steps (defining the training budget, budget=steps*128)

subset=experts # experts, 2domains, 3domains, 4domains

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
MIXTURE_TRESHOLD=102400 # Number of samples in training mixture for each domain, used to get the dataset name like expert_general-102400.
timelimit="03:00:00"
#timelimit="05:00:00"
# ----------------------------------------------------------------------------------------------- #

BENCHMARKS_CSV=$(IFS=, ; echo "${BENCHMARKS[*]}")
N=${#BENCHMARKS[@]}

# Ask user for confirmation
printf "Launching train jobs for: \nSubset: ${subset} \nModels: ${MODELS[*]} \nSFT strategies: ${SFT_STRATEGIES[*]} \nSteps: ${STEPS_LIST[*]} \nTimeLimit: ${timelimit} \n"
read -p "Are you sure? (enter): " confirmation
if [[ $confirmation != "" ]]; then
    exit 1
fi

submit_job () {
    jobid=$(sbatch --parsable "$@")
    echo "$jobid : $*" >> $LOGDIR/jobids.log
    echo "$jobid : $*" >> logs/joblogs/jobids_$(date +%Y-%m-%d).log
    echo $jobid

    # Dry run
    # echo "[DRY RUN] sbatch args: $*"
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

                    # Train expert
                    submit_job --job-name=train_expert_${DOMAIN} \
                        -t ${timelimit} \
                        --output=$LOGDIR/train_expert/%A_%x/%A_%x.out --error=$LOGDIR/train_expert/%A_%x/%A_%x.err \
                        scripts/experts/train_expert.sh ${BASE_MODEL} ${DOMAIN} ${SFT_STRATEGY} ${STEPS} ${MIXTURE_TRESHOLD} ${EXP_NAME}

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

                    # Train mixed model
                    submit_job --job-name=train_mixed_${DOM1}_${DOM2} \
                        -t ${timelimit} \
                        --output=$LOGDIR/train_mixed2/%A_%x/%A_%a_%x.out --error=$LOGDIR/train_mixed2/%A_%x/%A_%a_%x.err \
                        scripts/mixed2/train_mixed2.sh ${BASE_MODEL} ${DOM1} ${DOM2} ${SFT_STRATEGY} ${STEPS} ${MIXTURE_TRESHOLD} ${EXP_NAME}
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

                    # Train mixed model
                    submit_job --job-name=train_mixed_${DOM1}_${DOM2}_${DOM3} \
                        -t ${timelimit} \
                        --output=$LOGDIR/train_mixed3/%A_%x/%A_%a_%x.out --error=$LOGDIR/train_mixed3/%A_%x/%A_%a_%x.err \
                        scripts/mixed3/train_mixed3.sh ${BASE_MODEL} ${DOM1} ${DOM2} ${DOM3} ${SFT_STRATEGY} ${STEPS} ${MIXTURE_TRESHOLD} ${EXP_NAME}

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

                    # Train mixed model
                    submit_job --job-name=train_mixed_${DOM1}_${DOM2}_${DOM3}_${DOM4} \
                        -t ${timelimit} \
                        --output=$LOGDIR/train_mixed4/%A_%x/%A_%a_%x.out --error=$LOGDIR/train_mixed4/%A_%x/%A_%a_%x.err \
                        scripts/mixed4/train_mixed4.sh ${BASE_MODEL} ${DOM1} ${DOM2} ${DOM3} ${DOM4} ${SFT_STRATEGY} ${STEPS} ${MIXTURE_TRESHOLD} ${EXP_NAME}

                    done
                    
                done
                
            done
        done
    done
else
    echo "Unknown subset: ${subset}"
    exit 1
fi