#!/bin/bash

base_dir="/home/zuoyiyi/SNN/TCGA/DeepLIFT_results_2"

# 遍历每个癌症文件夹（如 BLCA、BRCA 等）
for cancer_dir in "$base_dir"/*; do
    if [ -d "$cancer_dir/feature_importance" ]; then
        old_dir="$cancer_dir/feature_importance"
        new_dir="$cancer_dir/deeplift_feature_importance"

        echo "重命名目录: $old_dir → $new_dir"
        #mv "$old_dir" "$new_dir"

        # 修改文件名：在每个文件名前加上 deeplift_
        for file in "$new_dir"/*.csv; do
            if [ -f "$file" ]; then
                old_name=$(basename "$file")
                new_name=$(echo "$old_name" | sed 's/_feature_importance_ranking/_deeplift_feature_importance_ranking/')
                #mv "$file" "$new_dir/$new_name"
                echo "重命名文件: $old_name → $new_name"
            fi
        done
    elif [ -d "$cancer_dir/deeplift_feature_importance" ]; then
        echo "检测到已存在目录: $cancer_dir/deeplift_feature_importance"
        for file in "$cancer_dir/deeplift_feature_importance"/*.csv; do
            if [ -f "$file" ]; then
                old_1_name=$(basename "$file")
                new_1_name=$(echo "$old_1_name" | sed 's/_feature_importance_ranking/_deeplift_feature_importance_ranking/')
                mv "$file" "$cancer_dir/deeplift_feature_importance/$new_1_name"
                echo "重命名文件: $old_1_name → $new_1_name"
            fi
        done
    fi
done
