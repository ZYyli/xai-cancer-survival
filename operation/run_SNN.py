import os
import subprocess

# 定义命令列表
commands = [
    "python main.py --csv_path /home/zuoyiyi/SNN/TCGA/datasets_csv/preprocess_1 --results_dir /home/zuoyiyi/SNN/TCGA/results_2/BLCA --cancer blca --keep_logs",
    "python main.py --csv_path /home/zuoyiyi/SNN/TCGA/datasets_csv/preprocess_1 --results_dir /home/zuoyiyi/SNN/TCGA/results_2/BRCA --cancer brca --keep_logs",
    "python main.py --csv_path /home/zuoyiyi/SNN/TCGA/datasets_csv/preprocess_1 --results_dir /home/zuoyiyi/SNN/TCGA/results_2/COADREAD --cancer coadread --keep_logs",
    "python main.py --csv_path /home/zuoyiyi/SNN/TCGA/datasets_csv/preprocess_1 --results_dir /home/zuoyiyi/SNN/TCGA/results_2/GBMLGG --cancer gbmlgg --keep_logs",
    "python main.py --csv_path /home/zuoyiyi/SNN/TCGA/datasets_csv/preprocess_1 --results_dir /home/zuoyiyi/SNN/TCGA/results_2/HNSC --cancer hnsc --keep_logs",
    "python main.py --csv_path /home/zuoyiyi/SNN/TCGA/datasets_csv/preprocess_1 --results_dir /home/zuoyiyi/SNN/TCGA/results_2/KIRC --cancer kirc --keep_logs",
    "python main.py --csv_path /home/zuoyiyi/SNN/TCGA/datasets_csv/preprocess_1 --results_dir /home/zuoyiyi/SNN/TCGA/results_2/KIRP --cancer kirp --keep_logs",
    "python main.py --csv_path /home/zuoyiyi/SNN/TCGA/datasets_csv/preprocess_1 --results_dir /home/zuoyiyi/SNN/TCGA/results_2/LGG --cancer lgg --keep_logs",
    "python main.py --csv_path /home/zuoyiyi/SNN/TCGA/datasets_csv/preprocess_1 --results_dir /home/zuoyiyi/SNN/TCGA/results_2/LIHC --cancer lihc --keep_logs",
    "python main.py --csv_path /home/zuoyiyi/SNN/TCGA/datasets_csv/preprocess_1 --results_dir /home/zuoyiyi/SNN/TCGA/results_2/LUAD --cancer luad --keep_logs",
    "python main.py --csv_path /home/zuoyiyi/SNN/TCGA/datasets_csv/preprocess_1 --results_dir /home/zuoyiyi/SNN/TCGA/results_2/LUSC --cancer lusc --keep_logs",
    "python main.py --csv_path /home/zuoyiyi/SNN/TCGA/datasets_csv/preprocess_1 --results_dir /home/zuoyiyi/SNN/TCGA/results_2/PAAD --cancer paad --keep_logs",
    "python main.py --csv_path /home/zuoyiyi/SNN/TCGA/datasets_csv/preprocess_1 --results_dir /home/zuoyiyi/SNN/TCGA/results_2/SKCM --cancer skcm --keep_logs",
    "python main.py --csv_path /home/zuoyiyi/SNN/TCGA/datasets_csv/preprocess_1 --results_dir /home/zuoyiyi/SNN/TCGA/results_2/STAD --cancer stad --keep_logs",
    "python main.py --csv_path /home/zuoyiyi/SNN/TCGA/datasets_csv/preprocess_1 --results_dir /home/zuoyiyi/SNN/TCGA/results_2/UCEC --cancer ucec --keep_logs"
]

# 遍历执行命令
for cmd in commands:
    print(f"\n执行命令: {cmd}")
    try:
        # 执行命令并捕获输出
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"命令成功执行，输出:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败，错误信息:\n{e.stderr}")