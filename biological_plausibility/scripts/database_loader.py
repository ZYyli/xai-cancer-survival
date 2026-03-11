#!/usr/bin/env python3
"""
数据库加载器 - 加载并解析各数据库 (支持四分类)

四分类定义:
- same_only: 基因仅在当前癌症类型有记录（无其他癌种）
- same_and_other: 基因在当前癌症类型有记录，同时也在其他癌种有记录
- other_only: 基因仅在其他癌症类型有记录（当前癌种无记录）
- not_supported: 基因在数据库中没有任何癌症关联记录
"""

import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATABASE_DIR

# TCGA 癌症类型到各数据库癌症名称的映射
# 每个 TCGA code 对应一个关键词列表，用于模糊匹配数据库中的癌症名称
TCGA_CANCER_MAPPING = {
    "BLCA": ["bladder", "urothelial"],
    "BRCA": ["breast"],
    "COADREAD": ["colorectal", "colon", "rectal", "bowel"],
    "GBMLGG": ["glioblastoma", "glioma", "brain"],
    "HNSC": ["head and neck", "oral", "laryn", "pharyn"],
    "KIRC": ["renal", "kidney", "clear cell"],
    "KIRP": ["renal", "kidney", "papillary"],
    "LGG": ["glioma", "low grade", "brain"],
    "LIHC": ["liver", "hepatocellular", "hepato"],
    "LUAD": ["lung adenocarcinoma", "lung adeno", "nsclc"],
    "LUSC": ["lung squamous", "lung", "nsclc"],
    "PAAD": ["pancrea"],
    "SKCM": ["melanoma", "skin"],
    "STAD": ["stomach", "gastric"],
    "UCEC": ["endometri", "uterine", "uterus"],
}


class DatabaseLoader:
    """加载并解析生物学数据库 (支持四分类)"""
    
    DGIDB_PRIORITY = {
        'approved_on_label': 4,
        'approved_off_label': 3,
        'clinical_trial': 2,
        'preclinical': 1,
    }

    def __init__(self):
        # 基因 -> 关联癌症列表 (用于四分类)
        self.oncokb_gene_cancers = {}  # {gene: set(cancer_types)}
        self.opentargets_gene_cancers = {}
        self.cancermine_gene_cancers = {}  # CancerMine: {gene: set(cancer_types)}
        
        # 兼容旧接口 (仅基因)
        self.oncokb_genes = {}
        self.dgidb_genes = {}
        self.cancermine_genes = {}  # CancerMine: {gene: role}
        
        self._load_all()
    
    def _load_all(self):
        """加载所有数据库"""
        self._load_oncokb()
        self._load_dgidb()
        self._load_opentargets()
        self._load_cancermine()
    
    def _load_oncokb(self):
        """
        加载 OncoKB 数据
        
        完整版 OncoKB 数据应包含 'oncokbAnnotated' 或 'tumorType' 字段
        如果有癌症类型字段，则解析为三分类格式
        """
        file_path = DATABASE_DIR / "oncokb_genes.tsv"
        if not file_path.exists():
            print(f"  ⚠ OncoKB 数据文件不存在: {file_path}")
            return
        
        df = pd.read_csv(file_path, sep="\t")
        
        # 检查是否有癌症类型字段
        cancer_col = None
        for col in ['tumorType', 'tumor_type', 'cancerType', 'cancer_type', 'Tumor Type']:
            if col in df.columns:
                cancer_col = col
                break
        
        # 解析基因和级别
        for _, row in df.iterrows():
            gene = row.get("hugoSymbol", row.get("Gene Symbol", ""))
            level = row.get("highestSensitiveLevel", row.get("Level", None))
            
            if gene and pd.notna(gene):
                gene = gene.upper()
                
                # 标准化 level
                if pd.notna(level):
                    level = str(level).replace("LEVEL_", "").upper()
                else:
                    level = None
                self.oncokb_genes[gene] = level
                
                # 解析癌症类型 (如果有)
                if cancer_col and pd.notna(row.get(cancer_col)):
                    cancer_type = str(row.get(cancer_col)).lower()
                    if gene not in self.oncokb_gene_cancers:
                        self.oncokb_gene_cancers[gene] = set()
                    self.oncokb_gene_cancers[gene].add(cancer_type)
                else:
                    # 没有癌症类型字段，标记为 "all" (泛癌)
                    if gene not in self.oncokb_gene_cancers:
                        self.oncokb_gene_cancers[gene] = {"all"}
        
        print(f"  ✓ 加载了 {len(self.oncokb_genes)} 个 OncoKB 基因")
    
    
    def _load_dgidb(self):
        """加载 DGIdb 数据"""
        file_path = DATABASE_DIR / "dgidb_interactions.tsv"
        if not file_path.exists():
            print(f"  ⚠ DGIdb 数据文件不存在: {file_path}")
            return
        
        df = pd.read_csv(file_path, sep="\t")
        
        for _, row in df.iterrows():
            gene = row.get("gene_name", row.get("gene", ""))
            source = str(row.get("drug_claim_source", row.get("source", ""))).lower()
            approved = row.get("approved", False)
            
            if gene and pd.notna(gene):
                gene = gene.upper()
                
                # 确定药物状态
                if "fda" in source or approved == True or approved == "True":
                    status = "approved_on_label"
                elif "clinical" in source or "trial" in source:
                    status = "clinical_trial"
                else:
                    status = "preclinical"
                
                # 保留最高级别
                current = self.dgidb_genes.get(gene)
                current_priority = self.DGIDB_PRIORITY.get(current, 0)
                status_priority = self.DGIDB_PRIORITY.get(status, 0)
                if current is None or status_priority > current_priority:
                    self.dgidb_genes[gene] = status
        
        print(f"  ✓ 加载了 {len(self.dgidb_genes)} 个 DGIdb 基因")
    
    
    def _load_opentargets(self):
        """
        加载 Open Targets 数据 (opentargets_associations.tsv)
        
        包含 tcga_code, gene_symbol, association_score 等字段
        """
        ot_file = DATABASE_DIR / "opentargets_associations.tsv"
        
        if not ot_file.exists():
            print(f"  ⚠ Open Targets 数据文件不存在: {ot_file}")
            return
        
        df = pd.read_csv(ot_file, sep="\t")
        
        for _, row in df.iterrows():
            gene = row.get("gene_symbol", "")
            tcga_code = row.get("tcga_code", "")
            
            if gene and pd.notna(gene):
                gene = gene.upper()
                if gene not in self.opentargets_gene_cancers:
                    self.opentargets_gene_cancers[gene] = set()
                
                if tcga_code and pd.notna(tcga_code):
                    self.opentargets_gene_cancers[gene].add(tcga_code.upper())
        
        print(f"  ✓ 加载了 {len(self.opentargets_gene_cancers)} 个 Open Targets 基因")
    
    def _load_cancermine(self):
        """
        加载 CancerMine 数据 (cancermine.tsv)
        
        CancerMine 是基于文献挖掘的癌症基因数据库，包含三种角色:
        - Driver: 驱动基因
        - Oncogene: 癌基因
        - Tumor_Suppressor: 抑癌基因
        
        数据格式: gene_normalized, cancer_normalized, role, citation_count
        """
        cm_file = DATABASE_DIR / "cancermine.tsv"
        
        if not cm_file.exists():
            print(f"  ⚠ CancerMine 数据文件不存在: {cm_file}")
            return
        
        df = pd.read_csv(cm_file, sep="\t")
        
        # 检测列名
        gene_col = 'gene_normalized' if 'gene_normalized' in df.columns else 'gene'
        cancer_col = 'cancer_normalized' if 'cancer_normalized' in df.columns else 'cancer'
        role_col = 'role' if 'role' in df.columns else None
        
        if gene_col not in df.columns:
            print(f"  ⚠ CancerMine 数据缺少基因列")
            return
        
        # 角色优先级 (用于保留最高优先级角色)
        role_priority = {'Driver': 3, 'Oncogene': 2, 'Tumor_Suppressor': 1}
        
        for _, row in df.iterrows():
            gene = row.get(gene_col, "")
            cancer = row.get(cancer_col, "")
            role = row.get(role_col, "Unknown") if role_col else "Unknown"
            
            if gene and pd.notna(gene):
                gene = gene.upper()
                
                # 保存角色 (保留最高优先级)
                current_role = self.cancermine_genes.get(gene)
                current_priority = role_priority.get(current_role, 0)
                new_priority = role_priority.get(role, 0)
                if current_role is None or new_priority > current_priority:
                    self.cancermine_genes[gene] = role
                
                # 保存癌症类型关联
                if cancer and pd.notna(cancer):
                    cancer_lower = str(cancer).lower()
                    if gene not in self.cancermine_gene_cancers:
                        self.cancermine_gene_cancers[gene] = set()
                    self.cancermine_gene_cancers[gene].add(cancer_lower)
        
        print(f"  ✓ 加载了 {len(self.cancermine_genes)} 个 CancerMine 基因")
    
    # ========================================================================
    # 四分类查询方法 (same_only / same_and_other / other_only / not_supported)
    # ========================================================================
    
    def _match_cancer(self, db_cancers: set, tcga_code: str) -> str:
        """
        判断基因的数据库癌症记录是否匹配 TCGA 癌症类型（四分类）
        
        Args:
            db_cancers: 数据库中该基因关联的癌症类型集合
            tcga_code: TCGA 癌症代码 (如 "BRCA")
        
        Returns:
            "same_only": 仅匹配当前癌症（无其他癌种）
            "same_and_other": 匹配当前癌症且有其他癌种
            "other_only": 仅匹配其他癌症（当前癌种无记录）
            "not_supported": 无癌症记录
        """
        if not db_cancers:
            return "not_supported"
        
        # 如果标记为 "all" (泛癌)，且只有这一个标记，视为 same_only
        if db_cancers == {"all"}:
            return "same_only"
        
        # 获取当前癌症的匹配关键词
        keywords = TCGA_CANCER_MAPPING.get(tcga_code.upper(), [])
        
        # 检查是否匹配当前癌症
        has_same = False
        has_other = False
        
        for cancer in db_cancers:
            if cancer == "all":
                has_same = True
                continue
            
            cancer_lower = cancer.lower()
            matched = False
            for keyword in keywords:
                if keyword.lower() in cancer_lower:
                    has_same = True
                    matched = True
                    break
            
            if not matched:
                has_other = True
        
        # 四分类逻辑
        if has_same and has_other:
            return "same_and_other"
        elif has_same and not has_other:
            return "same_only"
        elif not has_same and has_other:
            return "other_only"
        else:
            return "not_supported"
    
    def classify_oncokb(self, gene: str, tcga_code: str) -> str:
        """OncoKB 四分类"""
        gene = gene.upper()
        if gene not in self.oncokb_genes:
            return "not_supported"
        return self._match_cancer(self.oncokb_gene_cancers.get(gene, set()), tcga_code)
    
    def classify_dgidb(self, gene: str, tcga_code: str = None) -> str:
        """
        DGIdb 四分类 (DGIdb 是药物靶点数据库，与癌症类型关系较弱)
        
        只返回 "same_only" (有药物记录) 或 "not_supported" (无记录)
        """
        gene = gene.upper()
        if gene in self.dgidb_genes:
            return "same_only"  # 有药物记录，视为支持当前癌种
        return "not_supported"
    
    def classify_opentargets(self, gene: str, tcga_code: str) -> str:
        """
        Open Targets 四分类
        
        Open Targets 数据直接按 TCGA code 存储，可以精确匹配
        """
        gene = gene.upper()
        tcga_code = tcga_code.upper()
        
        if gene not in self.opentargets_gene_cancers:
            return "not_supported"
        
        cancer_set = self.opentargets_gene_cancers[gene]
        
        has_same = tcga_code in cancer_set
        has_other = len(cancer_set - {tcga_code}) > 0
        
        if has_same and has_other:
            return "same_and_other"
        elif has_same and not has_other:
            return "same_only"
        elif not has_same and has_other:
            return "other_only"
        else:
            return "not_supported"
    
    def classify_cancermine(self, gene: str, tcga_code: str) -> str:
        """
        CancerMine 四分类
        
        CancerMine 使用文献挖掘的癌症名称，需要模糊匹配 TCGA 癌症类型
        """
        gene = gene.upper()
        if gene not in self.cancermine_genes:
            return "not_supported"
        return self._match_cancer(self.cancermine_gene_cancers.get(gene, set()), tcga_code)
    
    def get_all_classifications(self, gene: str, tcga_code: str) -> dict:
        """
        获取基因在所有 5 个数据库中的四分类结果
        注意: DGIdb 无癌症类型信息，仅支持二分类
        
        Returns:
            {
                'oncokb': 'same_only' | 'same_and_other' | 'other_only' | 'not_supported',
                'dgidb': 'same_only' | 'not_supported' (仅二分类),
                'opentargets': ...,
                'cancermine': ...
            }
        """
        return {
            'oncokb': self.classify_oncokb(gene, tcga_code),
            'dgidb': self.classify_dgidb(gene, tcga_code),  # 仅二分类
            'opentargets': self.classify_opentargets(gene, tcga_code),
            'cancermine': self.classify_cancermine(gene, tcga_code),
        }
