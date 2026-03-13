set -euo pipefail

: "${TCGA_DIR:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

#数据预处理包括生成五折验证集和训练集
nohup python preprocessing_no_normalization.py > predata_no_nomalization_output 2>&1 &
nohup python preprocessing_cancer_single.py > predata_cancer_single_output 2>&1 &
#preprocess_1   五折数据
#preprocess_cancer_single   bootstrap需要的每种癌症所有样本

#SNN
nohup python run_SNN.py > SNN_2_output 2>&1 &    ##warmup=5, patience=7, stop_epoch=20
#results_2  五折，有早停

nohup python bootstrap_snn_evaluation.py > bootstrap_snn_evaluation_output 2>&1 &
#results_bootstrap  bootstrap结果 有早停


#作c-index图、生存曲线和动态AUC图
nohup python evaluate_plots_nested_cv.py > evaluate_nested_cv_fold_analysis_output 2>&1 &
##10 x 5-folds的三种图保存在results_nested_cv_plots_1文件夹中
nohup python bootstrap_boxplot_analysis.py --input_dir "$TCGA_DIR/results_bootstrap" --output_dir "$TCGA_DIR/bootstrap_boxplot" > bootstrap_boxplot_analysis_output 2>&1 &
##boostrap做了100次的c-index图，保存在bootstrap_boxplot文件夹中

#XAI+cox+correlation (DeepSHAP/SHAP/IG/LRP/PFI/DeepLIFT)
nohup python ig_individual_analysis.py --csv_path "$TCGA_DIR/datasets_csv/preprocess_1" --results_dir "$TCGA_DIR/results_2" --ig_dir "$TCGA_DIR/IG_results_2" > IG_individual_analysis_output 2>&1 &
nohup python lrp_individual_analysis.py --csv_path "$TCGA_DIR/datasets_csv/preprocess_1" --results_dir "$TCGA_DIR/results_2" --lrp_dir "$TCGA_DIR/LRP_results_2" > LRP_individual_analysis_output 2>&1 &
nohup python pfi_individual_analysis.py --csv_path "$TCGA_DIR/datasets_csv/preprocess_1" --results_dir "$TCGA_DIR/results_2" --pfi_dir "$TCGA_DIR/PFI_results_2" > PFI_individual_analysis_output 2>&1 &
nohup python shap_individual_analysis.py --csv_path "$TCGA_DIR/datasets_csv/preprocess_1" --results_dir "$TCGA_DIR/results_2" --shap_dir "$TCGA_DIR/shap_results_2" > shap_individual_analysis_output 2>&1 &
nohup python deepshap_individual_analysis.py --csv_path "$TCGA_DIR/datasets_csv/preprocess_1" --results_dir "$TCGA_DIR/results_2" --deepshap_dir "$TCGA_DIR/deepshap_results_2" > deepshap_individual_analysis_output 2>&1 &
nohup python deepLIFT_individual_analysis.py --csv_path "$TCGA_DIR/datasets_csv/preprocess_1" --results_dir "$TCGA_DIR/results_2" --deeplift_dir "$TCGA_DIR/DeepLIFT_results_2" > deepLIFT_individual_analysis_output 2>&1 &
#XAI+cox (KNN-CPI)
nohup python knn_cpi_individual_analysis.py --csv_path "$TCGA_DIR/datasets_csv/preprocess_1" --results_dir "$TCGA_DIR/results_2" --knn_cpi_dir "$TCGA_DIR/KNN_CPI_results_2" > knn_cpi_individual_analysis_output 2>&1 &

##bootstrap_xai 100次：
nohup python deepshap_bootstrap_analysis.py --csv_path "$TCGA_DIR/datasets_csv/preprocess_cancer_single" --results_dir "$TCGA_DIR/results_bootstrap" --deepshap_dir "$TCGA_DIR/deepshap_bootstrap_results" > deepshap_bootstrap_analysis_output 2>&1 &
nohup python shap_bootstrap_analysis.py --csv_path "$TCGA_DIR/datasets_csv/preprocess_cancer_single" --results_dir "$TCGA_DIR/results_bootstrap" --shap_dir "$TCGA_DIR/shap_bootstrap_results" > shap_bootstrap_analysis_output 2>&1 &
nohup python ig_bootstrap_analysis.py --csv_path "$TCGA_DIR/datasets_csv/preprocess_cancer_single" --results_dir "$TCGA_DIR/results_bootstrap" --ig_dir "$TCGA_DIR/IG_bootstrap_results" > ig_bootstrap_analysis_output 2>&1 &
nohup python lrp_bootstrap_analysis.py --csv_path "$TCGA_DIR/datasets_csv/preprocess_cancer_single" --results_dir "$TCGA_DIR/results_bootstrap" --lrp_dir "$TCGA_DIR/LRP_bootstrap_results" > lrp_bootstrap_analysis_output 2>&1 &
nohup python pfi_bootstrap_analysis.py --csv_path "$TCGA_DIR/datasets_csv/preprocess_cancer_single" --results_dir "$TCGA_DIR/results_bootstrap" --pfi_dir "$TCGA_DIR/PFI_bootstrap_results" > pfi_bootstrap_analysis_output 2>&1 &
nohup python deepLIFT_bootstrap_analysis.py --csv_path "$TCGA_DIR/datasets_csv/preprocess_cancer_single" --results_dir "$TCGA_DIR/results_bootstrap" --deeplift_dir "$TCGA_DIR/DeepLIFT_bootstrap_results" > deepLIFT_bootstrap_analysis_output 2>&1 &
nohup python knn_cpi_bootstrap_analysis.py --csv_path "$TCGA_DIR/datasets_csv/preprocess_cancer_single" --results_dir "$TCGA_DIR/results_bootstrap" --knn_cpi_dir "$TCGA_DIR/KNN_CPI_bootstrap_results" > knn_cpi_bootstrap_analysis_output 2>&1 &


###预后因子统计图（$TCGA_DIR/Prognostic_comparison_plots）
nohup python boxplot_prognotic.py > boxplot_prognotic_output 2>&1 &
nohup python corr_xai_cindex_heatmap.py > corr_xai_cindex_heatmap_output 2>&1 &


###特征稳定性分析
nohup python feature_stability_analysis.py > feature_stability_output_4 2>&1 &
nohup python feature_stability_analysis_bootstrap.py > feature_stability_bootstrap_output_100 2>&1 &

###稳定性统计图
nohup python corr_stability_cindex_heatmap.py > corr_stability_cindex_heatmap_output 2>&1 &
nohup python stability_visualization.py > stability_visualization_output 2>&1 &


##生物学合理性验证(4个数据库)
nohup python 01_download_databases.py > 01_download_databases_output 2>&1 &
nohup python "$TCGA_DIR/scripts/02_calculate_gene_scores.py" --all > 02_calculate_gene_scores_output 2>&1 &
nohup python "$TCGA_DIR/scripts/03_visualize_and_test.py" > 03_visualize_and_test_output 2>&1 &
nohup python "$TCGA_DIR/scripts/04_visualize_2.py" > 04_visualize_2_output 2>&1 &

#忠实度Fidelity
nohup python evaluate_faithfulness.py --project_root "$TCGA_DIR" --all_cancers > evaluate_faithfulness_output 2>&1 &