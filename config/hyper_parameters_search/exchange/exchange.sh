#!/bin/bash

# 1. 定义超参数搜索空间
BASE_PATCH_SIZES=(6)
ORI_PATCH_SIZES=(6)
HIDDEN_DIM=(64)
WAVELET_LEVEL=(1)
# 2. 这里的其他参数保持不变，提取出来方便修改
DATASET="exchange_rate_nips"
MAX_EPOCHS=50
DEVICE="[0]"
FLAG=1
# 3. 开始嵌套循环
for l in "${WAVELET_LEVEL[@]}"
do 
    for hid in "${HIDDEN_DIM[@]}"
    do
        for base in "${BASE_PATCH_SIZES[@]}"
        do
            for ori in "${ORI_PATCH_SIZES[@]}"
            do
                # 动态生成当前实验的根目录名称，例如: log_dir/ettm1_b4_o8
                EXP_DIR="log_dir/exchange-s/${DATASET}_h${hid}_b${base}_o${ori}_le${l}_seed_${FLAG}_gpu_${FLAG}"
                
                echo "=========================================================="
                echo "开始实验: Base Patch=$base, Ori Patch=$ori"
                echo "保存路径: $EXP_DIR"
                echo "当前时间: $(date)"
                echo "=========================================================="

                # 执行真正的 Python 命令
                # 注意：这里使用反斜杠 \ 来换行，保证命令的可读性
                python run.py --config config/stsf/exchange/tsflow_cond_ms.yaml \
                    --seed_everything "$FLAG" \
                    --data.data_manager.init_args.path datasets \
                    --trainer.default_root_dir "$EXP_DIR" \
                    --data.data_manager.init_args.split_val true \
                    --data.data_manager.init_args.dataset "$DATASET" \
                    --data.data_manager.init_args.context_length 30 \
                    --data.data_manager.init_args.prediction_length 30 \
                    --trainer.max_epochs "$MAX_EPOCHS" \
                    --trainer.devices "$DEVICE" \
                    --model.forecaster.init_args.base_patch_size "$base" \
                    --model.forecaster.init_args.ori_patch_size "$ori" \
                    --model.forecaster.init_args.wavelet_level "$l" \
                    --model.forecaster.init_args.hidden_dim "$hid" 

                echo "实验完成！"
                echo "----------------------------------------------------------"
            done
        done
    done
done