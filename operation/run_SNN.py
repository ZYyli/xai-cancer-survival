import os
import subprocess
from pathlib import Path

TCGA_DIR = Path(os.environ.get('TCGA_DIR', Path(__file__).resolve().parents[1])).resolve()

CSV_PATH = os.environ.get('CSV_PATH', str(TCGA_DIR / 'datasets_csv' / 'preprocess_1'))
RESULTS_ROOT = os.environ.get('RESULTS_ROOT', str(TCGA_DIR / 'results_2'))

CANCERS = [
    ('BLCA', 'blca'),
    ('BRCA', 'brca'),
    ('COADREAD', 'coadread'),
    ('GBMLGG', 'gbmlgg'),
    ('HNSC', 'hnsc'),
    ('KIRC', 'kirc'),
    ('KIRP', 'kirp'),
    ('LGG', 'lgg'),
    ('LIHC', 'lihc'),
    ('LUAD', 'luad'),
    ('LUSC', 'lusc'),
    ('PAAD', 'paad'),
    ('SKCM', 'skcm'),
    ('STAD', 'stad'),
    ('UCEC', 'ucec'),
]

# 定义命令列表
commands = [
    f"python main.py --csv_path {CSV_PATH} --results_dir {Path(RESULTS_ROOT) / cancer_dir} --cancer {cancer_arg} --keep_logs"
    for cancer_dir, cancer_arg in CANCERS
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