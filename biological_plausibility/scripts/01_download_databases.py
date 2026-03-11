#!/usr/bin/env python3
"""
Step 1: 下载并准备各数据库
- OncoKB: 癌症基因临床分级 (需API Token)
- CGC: Cancer Gene Census (需COSMIC账号)
- DGIdb: 药物-基因互作 (需手动下载)
- CIViC: 临床变异解读 (可自动下载)

运行方式:
    python scripts/01_download_databases.py
    python scripts/01_download_databases.py --skip_existing  # 跳过已存在的文件
"""

import os
import sys
import time
import argparse
import requests
import pandas as pd
from pathlib import Path
from io import StringIO

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATABASE_DIR, ONCOKB_API_TOKEN


def check_file_valid(filepath, min_rows=5):
    """检查文件是否存在且有效（非HTML，有足够行数）"""
    if not filepath.exists():
        return False
    try:
        # 检查是否为HTML（下载失败的情况）
        with open(filepath, 'r') as f:
            first_line = f.readline()
            if first_line.strip().startswith('<!') or first_line.strip().startswith('<html'):
                return False
        # 检查行数
        df = pd.read_csv(filepath, sep='\t' if filepath.suffix == '.tsv' else ',', nrows=min_rows)
        return len(df) >= min_rows
    except:
        return False


def write_version_file(data_file: Path, lines):
    """为数据文件生成/更新一个 *_VERSION.txt 说明文件

    版本文件命名规则: basename + "_VERSION.txt"，例如:
    - oncokb_genes.tsv -> oncokb_genes_VERSION.txt
    - opentargets_associations.tsv -> opentargets_associations_VERSION.txt
    """
    version_path = data_file.with_name(f"{data_file.stem}_VERSION.txt")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    header = [f"Generated: {timestamp}"]
    content = header + list(lines)
    try:
        with open(version_path, "w", encoding="utf-8") as f:
            for line in content:
                f.write(str(line).rstrip("\n") + "\n")
    except Exception as e:
        print(f"  ⚠ 写入版本说明文件失败: {version_path} ({e})")

def download_oncokb():
    """
    下载 OncoKB 癌症基因列表
    
    注意: OncoKB 需要 API token，请先申请:
    https://www.oncokb.org/account/register
    """
    print("\n" + "="*60)
    print("下载 OncoKB 数据")
    print("="*60)
    
    output_file = DATABASE_DIR / "oncokb_genes.tsv"
    
    if ONCOKB_API_TOKEN:
        # 使用 API 获取完整数据
        headers = {"Authorization": f"Bearer {ONCOKB_API_TOKEN}"}
        url = "https://www.oncokb.org/api/v1/utils/allCuratedGenes"
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            genes = response.json()
            
            df = pd.DataFrame(genes)
            df.to_csv(output_file, sep="\t", index=False)
            print(f"✓ 已保存 {len(df)} 个 OncoKB 基因到 {output_file}")
            write_version_file(output_file, [
                f"Source: {url}",
                "Description: OncoKB curated genes downloaded via official API",
            ])
            return True
        except Exception as e:
            print(f"✗ OncoKB API 请求失败: {e}")
    
    # 如果没有 API token，创建模板文件
    print("⚠ 未设置 ONCOKB_API_TOKEN 环境变量")
    print("  请手动下载数据或设置环境变量后重试")
    print("  申请地址: https://www.oncokb.org/account/register")
    
    # 创建模板文件，包含常见癌症基因
    template_genes = [
        {"hugoSymbol": "TP53", "highestSensitiveLevel": "1", "oncogene": False, "tsg": True},
        {"hugoSymbol": "EGFR", "highestSensitiveLevel": "1", "oncogene": True, "tsg": False},
        {"hugoSymbol": "BRAF", "highestSensitiveLevel": "1", "oncogene": True, "tsg": False},
        {"hugoSymbol": "KRAS", "highestSensitiveLevel": "1", "oncogene": True, "tsg": False},
        {"hugoSymbol": "PIK3CA", "highestSensitiveLevel": "1", "oncogene": True, "tsg": False},
        {"hugoSymbol": "ERBB2", "highestSensitiveLevel": "1", "oncogene": True, "tsg": False},
        {"hugoSymbol": "ALK", "highestSensitiveLevel": "1", "oncogene": True, "tsg": False},
        {"hugoSymbol": "ROS1", "highestSensitiveLevel": "1", "oncogene": True, "tsg": False},
        {"hugoSymbol": "MET", "highestSensitiveLevel": "1", "oncogene": True, "tsg": False},
        {"hugoSymbol": "RET", "highestSensitiveLevel": "1", "oncogene": True, "tsg": False},
        {"hugoSymbol": "BRCA1", "highestSensitiveLevel": "1", "oncogene": False, "tsg": True},
        {"hugoSymbol": "BRCA2", "highestSensitiveLevel": "1", "oncogene": False, "tsg": True},
        {"hugoSymbol": "ATM", "highestSensitiveLevel": "2", "oncogene": False, "tsg": True},
        {"hugoSymbol": "PTEN", "highestSensitiveLevel": "2", "oncogene": False, "tsg": True},
        {"hugoSymbol": "AKT1", "highestSensitiveLevel": "2", "oncogene": True, "tsg": False},
    ]
    
    print("\n  ⚠ 创建模板文件（仅包含示例基因，请替换为完整数据）")
    df = pd.DataFrame(template_genes)
    df.to_csv(output_file, sep="\t", index=False)
    print(f"  模板已保存到 {output_file}")
    write_version_file(output_file, [
        "Source: local template (no ONCOKB_API_TOKEN)",
        "Description: example curated genes only, please replace with full OncoKB data",
        "Manual action: download full data from https://www.oncokb.org/ and overwrite this file",
    ])
    
    return False


def download_dgidb():
    """
    下载 DGIdb 药物-基因互作数据
    
    注意: DGIdb 网站已改版 (2024)，旧API失效，需手动下载
    下载地址: https://dgidb.org/downloads
    """
    print("\n" + "="*60)
    print("下载 DGIdb 数据")
    print("="*60)
    
    output_file = DATABASE_DIR / "dgidb_interactions.tsv"
    
    # 检查是否已有有效文件
    if check_file_valid(output_file, min_rows=100):
        print(f"  ✓ 已存在有效数据文件: {output_file}")
        df = pd.read_csv(output_file, sep="\t")
        print(f"    包含 {len(df)} 条记录")
        # 保留用户手动下载版本，不覆盖版本说明，只在缺失时创建一个简短说明
        if not (output_file.parent / f"{output_file.stem}_VERSION.txt").exists():
            write_version_file(output_file, [
                "Source: user-provided DGIdb interactions.tsv",
                "Description: manually downloaded from https://dgidb.org/downloads",
            ])
        return True
    
    # DGIdb 网站已改版，API不可用，需要手动下载
    print("  ⚠ DGIdb 网站已改版，需要手动下载数据")
    print("")
    print("  【手动下载步骤】:")
    print("    1. 浏览器访问: https://dgidb.org/downloads")
    print("    2. 找到并下载: interactions.tsv")
    print("    3. 将文件保存到:")
    print(f"       {output_file}")
    print("")
    
    # 创建扩展模板（包含常见 FDA 批准靶点药物）
    print("  创建临时模板文件（包含常见靶点药物）...")
    
    template = [
        # EGFR 抑制剂
        {"gene_name": "EGFR", "drug_name": "Erlotinib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "EGFR", "drug_name": "Gefitinib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "EGFR", "drug_name": "Osimertinib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "EGFR", "drug_name": "Afatinib", "interaction_types": "inhibitor", "approved": True},
        # BRAF 抑制剂
        {"gene_name": "BRAF", "drug_name": "Vemurafenib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "BRAF", "drug_name": "Dabrafenib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "BRAF", "drug_name": "Encorafenib", "interaction_types": "inhibitor", "approved": True},
        # HER2/ERBB2 抑制剂
        {"gene_name": "ERBB2", "drug_name": "Trastuzumab", "interaction_types": "antibody", "approved": True},
        {"gene_name": "ERBB2", "drug_name": "Pertuzumab", "interaction_types": "antibody", "approved": True},
        {"gene_name": "ERBB2", "drug_name": "Lapatinib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "ERBB2", "drug_name": "Neratinib", "interaction_types": "inhibitor", "approved": True},
        # ALK 抑制剂
        {"gene_name": "ALK", "drug_name": "Crizotinib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "ALK", "drug_name": "Alectinib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "ALK", "drug_name": "Brigatinib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "ALK", "drug_name": "Lorlatinib", "interaction_types": "inhibitor", "approved": True},
        # PARP 抑制剂 (BRCA)
        {"gene_name": "BRCA1", "drug_name": "Olaparib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "BRCA2", "drug_name": "Olaparib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "BRCA1", "drug_name": "Rucaparib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "BRCA2", "drug_name": "Rucaparib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "BRCA1", "drug_name": "Niraparib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "BRCA2", "drug_name": "Niraparib", "interaction_types": "inhibitor", "approved": True},
        # PIK3CA
        {"gene_name": "PIK3CA", "drug_name": "Alpelisib", "interaction_types": "inhibitor", "approved": True},
        # KRAS
        {"gene_name": "KRAS", "drug_name": "Sotorasib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "KRAS", "drug_name": "Adagrasib", "interaction_types": "inhibitor", "approved": True},
        # RET
        {"gene_name": "RET", "drug_name": "Selpercatinib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "RET", "drug_name": "Pralsetinib", "interaction_types": "inhibitor", "approved": True},
        # MET
        {"gene_name": "MET", "drug_name": "Capmatinib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "MET", "drug_name": "Tepotinib", "interaction_types": "inhibitor", "approved": True},
        # ROS1
        {"gene_name": "ROS1", "drug_name": "Crizotinib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "ROS1", "drug_name": "Entrectinib", "interaction_types": "inhibitor", "approved": True},
        # NTRK
        {"gene_name": "NTRK1", "drug_name": "Larotrectinib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "NTRK2", "drug_name": "Larotrectinib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "NTRK3", "drug_name": "Larotrectinib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "NTRK1", "drug_name": "Entrectinib", "interaction_types": "inhibitor", "approved": True},
        # FGFR
        {"gene_name": "FGFR2", "drug_name": "Pemigatinib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "FGFR2", "drug_name": "Erdafitinib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "FGFR3", "drug_name": "Erdafitinib", "interaction_types": "inhibitor", "approved": True},
        # IDH
        {"gene_name": "IDH1", "drug_name": "Ivosidenib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "IDH2", "drug_name": "Enasidenib", "interaction_types": "inhibitor", "approved": True},
        # BCR-ABL
        {"gene_name": "BCR", "drug_name": "Imatinib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "ABL1", "drug_name": "Imatinib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "ABL1", "drug_name": "Dasatinib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "ABL1", "drug_name": "Nilotinib", "interaction_types": "inhibitor", "approved": True},
        # KIT
        {"gene_name": "KIT", "drug_name": "Imatinib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "KIT", "drug_name": "Sunitinib", "interaction_types": "inhibitor", "approved": True},
        # PDGFRA
        {"gene_name": "PDGFRA", "drug_name": "Imatinib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "PDGFRA", "drug_name": "Avapritinib", "interaction_types": "inhibitor", "approved": True},
        # FLT3
        {"gene_name": "FLT3", "drug_name": "Midostaurin", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "FLT3", "drug_name": "Gilteritinib", "interaction_types": "inhibitor", "approved": True},
        # BTK
        {"gene_name": "BTK", "drug_name": "Ibrutinib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "BTK", "drug_name": "Acalabrutinib", "interaction_types": "inhibitor", "approved": True},
        # BCL2
        {"gene_name": "BCL2", "drug_name": "Venetoclax", "interaction_types": "inhibitor", "approved": True},
        # VEGFR
        {"gene_name": "KDR", "drug_name": "Bevacizumab", "interaction_types": "antibody", "approved": True},
        {"gene_name": "KDR", "drug_name": "Sorafenib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "KDR", "drug_name": "Sunitinib", "interaction_types": "inhibitor", "approved": True},
        # ESR1 (雌激素受体)
        {"gene_name": "ESR1", "drug_name": "Tamoxifen", "interaction_types": "antagonist", "approved": True},
        {"gene_name": "ESR1", "drug_name": "Fulvestrant", "interaction_types": "antagonist", "approved": True},
        # AR (雄激素受体)
        {"gene_name": "AR", "drug_name": "Enzalutamide", "interaction_types": "antagonist", "approved": True},
        {"gene_name": "AR", "drug_name": "Abiraterone", "interaction_types": "inhibitor", "approved": True},
        # CDK4/6
        {"gene_name": "CDK4", "drug_name": "Palbociclib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "CDK6", "drug_name": "Palbociclib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "CDK4", "drug_name": "Ribociclib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "CDK6", "drug_name": "Ribociclib", "interaction_types": "inhibitor", "approved": True},
        # MTOR
        {"gene_name": "MTOR", "drug_name": "Everolimus", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "MTOR", "drug_name": "Temsirolimus", "interaction_types": "inhibitor", "approved": True},
        # SMO (Hedgehog)
        {"gene_name": "SMO", "drug_name": "Vismodegib", "interaction_types": "inhibitor", "approved": True},
        {"gene_name": "SMO", "drug_name": "Sonidegib", "interaction_types": "inhibitor", "approved": True},
    ]
    
    df = pd.DataFrame(template)
    df.to_csv(output_file, sep="\t", index=False)
    print(f"  ✓ 模板已保存: {output_file}")
    write_version_file(output_file, [
        "Source: local template (DGIdb manual download required)",
        "Description: contains common FDA-approved targeted agents only, please replace with full DGIdb interactions.tsv",
        "Manual action: download interactions.tsv from https://dgidb.org/downloads and overwrite this file",
    ])
    print(f"    包含 {len(template)} 条 FDA 批准靶向药物记录")
    print("  ⚠ 请尽快下载完整数据替换此模板")
    return False


def download_opentargets():
    """
    下载 Open Targets Platform 基因-疾病关联数据
    
    Open Targets API 免费无需注册
    为每个 TCGA 癌症类型下载相关基因列表
    """
    print("\n" + "="*60)
    print("下载 Open Targets 数据")
    print("="*60)
    
    output_file = DATABASE_DIR / "opentargets_associations.tsv"
    
    # 检查是否已有有效文件
    if check_file_valid(output_file, min_rows=100):
        print(f"  ✓ 已存在有效数据文件: {output_file}")
        df = pd.read_csv(output_file, sep="\t")
        print(f"    包含 {len(df)} 条记录")
        if not (output_file.parent / f"{output_file.stem}_VERSION.txt").exists():
            write_version_file(output_file, [
                "Source: user-provided or previously downloaded Open Targets associations",
                "Description: gene-disease associations for TCGA cancer types via EFO/MONDO IDs",
                "URL: https://api.platform.opentargets.org/api/v4/graphql",
            ])
        return True
    
    # TCGA 癌种到 Open Targets EFO/MONDO ID 的映射 (已通过 API 验证)
    TCGA_TO_EFO = {
        "BLCA": ("MONDO_0001187", "bladder carcinoma"),
        "BRCA": ("EFO_0000305", "breast carcinoma"),
        "COADREAD": ("MONDO_0005575", "colorectal cancer"),
        "GBMLGG": ("EFO_0000519", "glioblastoma"),
        "HNSC": ("EFO_0000181", "head and neck squamous cell carcinoma"),
        "KIRC": ("EFO_0000349", "renal cell carcinoma"),
        "KIRP": ("EFO_0000640", "papillary renal cell carcinoma"),
        "LGG": ("EFO_0005543", "low grade glioma"),
        "LIHC": ("EFO_0000182", "hepatocellular carcinoma"),
        "LUAD": ("EFO_0000571", "lung adenocarcinoma"),
        "LUSC": ("EFO_0000708", "lung squamous cell carcinoma"),
        "PAAD": ("EFO_1000044", "pancreatic adenocarcinoma"),
        "SKCM": ("EFO_0000389", "melanoma"),
        "STAD": ("EFO_0000503", "stomach carcinoma"),
        "UCEC": ("EFO_1001512", "endometrial carcinoma"),
    }
    
    # API 分页限制
    MAX_PAGE_SIZE = 2500  # API 最大允许 3000，保守设置 2500
    
    GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
    
    # GraphQL 查询 - 获取疾病关联的所有靶点
    query = """
    query DiseaseAssociations($efoId: String!, $size: Int!) {
        disease(efoId: $efoId) {
            id
            name
            associatedTargets(page: { index: 0, size: $size }) {
                count
                rows {
                    target {
                        id
                        approvedSymbol
                    }
                    score
                }
            }
        }
    }
    """
    
    all_associations = []
    
    print(f"  将下载 {len(TCGA_TO_EFO)} 种癌症类型的关联数据...")
    
    for tcga_code, (efo_id, disease_name) in TCGA_TO_EFO.items():
        print(f"    {tcga_code} ({disease_name})...", end=" ", flush=True)
        
        try:
            # 分页下载关联数据
            page_index = 0
            total_rows = 0
            
            while True:
                response = requests.post(
                    GRAPHQL_URL,
                    json={
                        "query": query,
                        "variables": {"efoId": efo_id, "size": MAX_PAGE_SIZE}
                    },
                    timeout=120,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                data = response.json()
                
                # 检查是否有错误
                if "errors" in data:
                    error_msg = data['errors'][0].get('message', '')[:100]
                    print(f"API错误: {error_msg}")
                    break
                
                # 安全解析
                disease_data = data.get("data", {})
                if disease_data is None:
                    print("data 为空")
                    break
                
                disease_info = disease_data.get("disease")
                if disease_info is None:
                    print(f"未找到疾病 {efo_id}")
                    break
                
                associated = disease_info.get("associatedTargets")
                if associated is None:
                    print("无关联靶点")
                    break
                
                rows = associated.get("rows", [])
                if not rows:
                    break
                
                for row in rows:
                    target = row.get("target", {}) or {}
                    gene_symbol = target.get("approvedSymbol", "")
                    if gene_symbol:
                        all_associations.append({
                            "tcga_code": tcga_code,
                            "efo_id": efo_id,
                            "disease_name": disease_name,
                            "gene_symbol": gene_symbol,
                            "ensembl_id": target.get("id", ""),
                            "association_score": row.get("score", 0),
                        })
                
                total_rows += len(rows)
                
                # 只取第一页 (2500 个基因已经足够覆盖 Top 100)
                break
            
            if total_rows > 0:
                print(f"{total_rows} 个基因")
            
            # 避免 API 限流
            time.sleep(0.3)
            
        except requests.exceptions.RequestException as e:
            print(f"网络错误: {e}")
        except Exception as e:
            print(f"失败: {type(e).__name__}: {e}")
    
    if all_associations:
        df = pd.DataFrame(all_associations)
        df.to_csv(output_file, sep="\t", index=False)
        print(f"\n  ✓ 已保存 {len(df)} 条关联记录到 {output_file}")
        
        # 统计
        gene_count = df['gene_symbol'].nunique()
        print(f"    涵盖 {gene_count} 个独立基因")
        write_version_file(output_file, [
            "Source: Open Targets Platform GraphQL API",
            "Description: gene-disease associations for TCGA cancer types (first page up to 2500 targets per disease)",
            "URL: https://api.platform.opentargets.org/api/v4/graphql",
        ])
        return True
    else:
        print("  ✗ 未能下载任何数据")
        return False


def download_cancermine():
    """
    下载 CancerMine 文献挖掘数据
    
    CancerMine 是基于文献挖掘的癌症基因数据库，包含三种角色:
    - Driver: 驱动基因
    - Oncogene: 癌基因  
    - Tumor_Suppressor: 抑癌基因
    
    数据下载地址: https://zenodo.org/records/7689627
    (原网站 http://bionlp.bcgsc.ca/cancermine/ 已下线)
    """
    print("\n" + "="*60)
    print("下载 CancerMine 数据")
    print("="*60)
    
    output_file = DATABASE_DIR / "cancermine.tsv"
    
    # 检查是否已有有效文件
    if check_file_valid(output_file, min_rows=100):
        print(f"  ✓ 已存在有效数据文件: {output_file}")
        df = pd.read_csv(output_file, sep="\t")
        print(f"    包含 {len(df)} 条记录")
        if not (output_file.parent / f"{output_file.stem}_VERSION.txt").exists():
            write_version_file(output_file, [
                "Source: user-provided or previously downloaded CancerMine data",
                "Description: literature-mined cancer gene associations (Driver/Oncogene/Tumor_Suppressor)",
                "URL: https://zenodo.org/records/7689627",
            ])
        return True
    
    # CancerMine 公开下载链接 (Zenodo - 原网站已下线)
    url = "https://zenodo.org/records/7689627/files/cancermine_collated.tsv?download=1"
    
    try:
        print(f"  下载: {url}")
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        
        # 检查是否为有效TSV
        if response.text.strip().startswith('<!') or '<html' in response.text[:100].lower():
            print("    ✗ 返回HTML而非数据")
            raise Exception("Invalid response")
        
        df = pd.read_csv(StringIO(response.text), sep="\t")
        if len(df) > 100:
            df.to_csv(output_file, sep="\t", index=False)
            print(f"  ✓ 已保存 {len(df)} 条 CancerMine 记录到 {output_file}")
            
            # 统计角色分布
            if 'role' in df.columns:
                role_counts = df['role'].value_counts()
                print(f"    角色分布:")
                for role, count in role_counts.items():
                    print(f"      - {role}: {count}")
            
            # 统计基因和癌症数量
            gene_col = 'gene_normalized' if 'gene_normalized' in df.columns else 'gene'
            cancer_col = 'cancer_normalized' if 'cancer_normalized' in df.columns else 'cancer'
            if gene_col in df.columns:
                print(f"    独立基因数: {df[gene_col].nunique()}")
            if cancer_col in df.columns:
                print(f"    独立癌症类型数: {df[cancer_col].nunique()}")
            
            write_version_file(output_file, [
                f"Source: {url}",
                "Description: CancerMine collated data - literature-mined cancer gene associations",
                "Roles: Driver, Oncogene, Tumor_Suppressor",
                f"Total records: {len(df)}",
            ])
            return True
        else:
            print("    ✗ 数据行数不足")
            
    except Exception as e:
        print(f"  ✗ 下载失败: {e}")
    
    # 下载失败，创建模板
    print("  ⚠ 自动下载失败，创建模板文件...")
    print("  请手动下载: https://zenodo.org/records/7689627")
    print("  选择 cancermine_collated.tsv 并保存为 databases/cancermine.tsv")
    
    template = [
        {"gene_normalized": "TP53", "cancer_normalized": "breast cancer", "role": "Tumor_Suppressor", "citation_count": 100},
        {"gene_normalized": "BRCA1", "cancer_normalized": "breast cancer", "role": "Tumor_Suppressor", "citation_count": 80},
        {"gene_normalized": "BRCA2", "cancer_normalized": "breast cancer", "role": "Tumor_Suppressor", "citation_count": 70},
        {"gene_normalized": "EGFR", "cancer_normalized": "lung cancer", "role": "Oncogene", "citation_count": 90},
        {"gene_normalized": "KRAS", "cancer_normalized": "colorectal cancer", "role": "Driver", "citation_count": 85},
        {"gene_normalized": "BRAF", "cancer_normalized": "melanoma", "role": "Oncogene", "citation_count": 75},
        {"gene_normalized": "PIK3CA", "cancer_normalized": "breast cancer", "role": "Oncogene", "citation_count": 60},
        {"gene_normalized": "PTEN", "cancer_normalized": "prostate cancer", "role": "Tumor_Suppressor", "citation_count": 55},
        {"gene_normalized": "APC", "cancer_normalized": "colorectal cancer", "role": "Tumor_Suppressor", "citation_count": 50},
        {"gene_normalized": "IDH1", "cancer_normalized": "glioma", "role": "Driver", "citation_count": 45},
        {"gene_normalized": "ERBB2", "cancer_normalized": "breast cancer", "role": "Oncogene", "citation_count": 65},
        {"gene_normalized": "MYC", "cancer_normalized": "lymphoma", "role": "Oncogene", "citation_count": 70},
        {"gene_normalized": "RB1", "cancer_normalized": "retinoblastoma", "role": "Tumor_Suppressor", "citation_count": 40},
        {"gene_normalized": "VHL", "cancer_normalized": "kidney cancer", "role": "Tumor_Suppressor", "citation_count": 35},
        {"gene_normalized": "ALK", "cancer_normalized": "lung cancer", "role": "Oncogene", "citation_count": 55},
    ]
    df = pd.DataFrame(template)
    df.to_csv(output_file, sep="\t", index=False)
    print(f"  模板已保存到 {output_file}")
    write_version_file(output_file, [
        "Source: local template (CancerMine manual download required)",
        "Description: example cancer gene associations only, please replace with full CancerMine data",
        "Manual action: download cancermine_collated.tsv from http://bionlp.bcgsc.ca/cancermine/ and overwrite this file",
    ])
    return False


def main():
    parser = argparse.ArgumentParser(description="下载生物学数据库")
    parser.add_argument('--skip_existing', action='store_true', 
                        help='跳过已存在的有效文件')
    args = parser.parse_args()
    
    print("="*60)
    print("XAI 生物学合理性评估 - 数据库下载")
    print("="*60)
    
    # 确保目录存在
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"数据库目录: {DATABASE_DIR}")
    
    # 下载各数据库
    results = {
        "OncoKB": download_oncokb(),
        "DGIdb": download_dgidb(),
        "OpenTargets": download_opentargets(),
        "CancerMine": download_cancermine()
    }
    
    # 汇总
    print("\n" + "="*60)
    print("下载汇总")
    print("="*60)
    
    success_count = 0
    for db, success in results.items():
        if success:
            status = "✓ 完整数据"
            success_count += 1
        else:
            status = "⚠ 模板数据 (需替换)"
        print(f"  {db}: {status}")
    
    print(f"\n  完整数据: {success_count}/6")
    
    # 检查文件大小
    print("\n" + "="*60)
    print("文件详情")
    print("="*60)
    
    files_info = [
        ("oncokb_genes.tsv", "OncoKB"),
        ("dgidb_interactions.tsv", "DGIdb"),
        ("opentargets_associations.tsv", "OpenTargets"),
        ("cancermine.tsv", "CancerMine"),
    ]
    
    for filename, db_name in files_info:
        filepath = DATABASE_DIR / filename
        if filepath.exists():
            try:
                sep = '\t' if filename.endswith('.tsv') else ','
                df = pd.read_csv(filepath, sep=sep)
                print(f"  {db_name}: {len(df)} 条记录")
            except:
                print(f"  {db_name}: 文件存在但无法读取")
        else:
            print(f"  {db_name}: 文件不存在")
    
    # 下一步提示
    print("\n" + "="*60)
    print("下一步")
    print("="*60)
    
    if success_count < 6:
        print("  需要补充的数据:")
        if not results["OncoKB"]:
            print("    - OncoKB: 申请 API Token → https://www.oncokb.org/account/register")
        if not results["DGIdb"]:
            print("    - DGIdb: 下载数据 → https://dgidb.org/downloads")
        if not results["OpenTargets"]:
            print("    - OpenTargets: 重新运行脚本 (API 免费)")
        if not results["CancerMine"]:
            print("    - CancerMine: 下载数据 → https://zenodo.org/records/7689627")
        print("")
    
    print("  继续下一步 (可使用模板数据测试):")
    print("    python scripts/02_calculate_gene_scores.py --cancer BRCA --xai DeepLIFT")


if __name__ == "__main__":
    main()
