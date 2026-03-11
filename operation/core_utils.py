from argparse import Namespace
import os
import numpy as np
import random
from sksurv.metrics import concordance_index_censored
import torch
import torch.optim as optim
from model_genomic import SNN
from loss_func import NLLSurvLoss
from utils import l1_reg_omic, print_network

import torch

class EarlyStoppingWithCIndex:
    """
    Early stopping based on validation loss, 
    but saving best model according to C-index (with warmup and loss filtering).
    warmup 阶段：完全不保存 loss 模型，只等训练稳定后再记录。
    warmup 后：每出现新的最优 loss 都保存对应模型。
    C-index 优先：只要 loss 没大幅下降，C-index 优先保存。
    早停触发：不会丢失 warmup 后的 loss 最优模型，也不依赖最后一轮。
    """
    def __init__(self, warmup=5, patience=7, stop_epoch=20, verbose=False, ckpt_name="best_cindex_checkpoint.pt", fold=None, results_dir=None, repeat=None):
        """
        Args:
            warmup (int): 最小训练轮数，前 warmup 个 epoch 不会触发保存/早停
            patience (int): 当验证 loss 多次不下降时，允许的等待次数
            stop_epoch (int): 只有 epoch >= stop_epoch 才允许早停
            verbose (bool): 是否打印日志
            ckpt_name (str): 保存模型的文件名
            fold (int): 当前fold编号
            results_dir (str): 结果保存目录
            repeat (int): 当前重复实验编号
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.ckpt_name = ckpt_name
        self.fold = fold
        self.results_dir = results_dir
        self.repeat = repeat

        self.counter = 0
        self.best_loss = None           # warmup 后记录的最优 loss
        self.best_loss_epoch = None
        self.best_cindex = None
        self.best_cindex_epoch = None
        self.early_stop = False

    def __call__(self, epoch, val_loss, val_cindex, model):
        """
        Args:
            epoch (int): 当前训练轮数
            val_loss (float): 验证集 loss
            val_cindex (float): 验证集 C-index
            model (torch.nn.Module): 当前模型
        """
        # ---------- warmup 阶段 ----------
        if epoch < self.warmup:
            return  # warmup 阶段不记录、不保存

        # --------- 早停逻辑（基于 loss） ---------
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_loss_epoch = epoch
            self.counter = 0
            # 保存 loss 最优模型
            self.save_checkpoint(model, best_cindex=False)
            if self.verbose:
                print(f"Epoch {epoch}: Saved new best-loss model with loss={val_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience and epoch >= self.stop_epoch:
                self.early_stop = True
                if self.verbose:
                    best_loss_str = f"{self.best_loss:.4f}" if self.best_loss is not None else "None"
                    best_cindex_str = f"{self.best_cindex:.4f}" if self.best_cindex is not None else "None"
                    print(f"Early stopping at epoch {epoch}, best val_loss={best_loss_str}, "
                          f"best val_cindex={best_cindex_str} (epoch {self.best_cindex_epoch})")

        # 保存逻辑：C-index 优先
        # 仅当 val_loss 不比历史最优差太多时才保存 C-index 模型
        if val_cindex is not None and (self.best_cindex is None or val_cindex > self.best_cindex) and val_loss <= self.best_loss * 1.2:
            self.best_cindex = val_cindex
            self.best_cindex_epoch = epoch
            self.save_checkpoint(model, best_cindex=True)
            if self.verbose:
                print(f"Epoch {epoch}: Saved new best model with C-index={val_cindex:.4f}, loss={val_loss:.4f}")

    def save_checkpoint(self, model, best_cindex=True):
        if self.fold is not None and self.results_dir is not None:
            # 包含repeat编号以避免不同重复实验的文件覆盖
            if self.repeat is not None:
                fname = f"repeat{self.repeat}_s_{self.fold}_{'best_cindex' if best_cindex else 'best_loss'}_checkpoint.pt"
            else:
                fname = f"s_{self.fold}_{'best_cindex' if best_cindex else 'best_loss'}_checkpoint.pt"
            path = os.path.join(self.results_dir, fname)
        else:
            path = "best_cindex_checkpoint.pt" if best_cindex else "best_loss_checkpoint.pt"
        torch.save(model.state_dict(), path)

def train(train_loader, val_loader, test_loader, cur: int, args: Namespace, seed=None, repeat=None):
    """
    训练函数，支持嵌套交叉验证
    
    Args:
        train_loader: 用于训练的数据加载器
        val_loader: 用于早停和模型选择的数据加载器  
        test_loader: 用于最终评估的数据加载器
        cur: 当前fold编号
        args: 参数命名空间
        seed: 随机种子
        repeat: 当前重复实验编号
    
    Returns:
        test_results: 测试集上的详细结果
        test_cindex: 测试集上的C-index
    """
    print('\n开始训练...')

    # 设置随机种子
    if seed is not None:
        args.seed = seed
    elif not hasattr(args, 'seed'):
        # 从 seeds 参数中推导当前 seed（支持列表或字符串格式）
        if hasattr(args, 'seeds'):
            if isinstance(args.seeds, list):
                args.seed = args.seeds[0]  # 使用第一个种子
            else:
                # 解析字符串格式的 seeds（如 "1-5" 或 "1,3,5"）
                first_seed = args.seeds.split(',')[0]
                if '-' in first_seed:
                    args.seed = int(first_seed.split('-')[0])
                else:
                    args.seed = int(first_seed)
        else:
            args.seed = 1  # 默认种子

    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.makedirs(writer_dir, exist_ok=True)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None

    print('初始化损失函数...', end=' ')
    loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    print('完成!')

    # 设置手动正则化函数 (L1 正则化)
    reg_fn = l1_reg_omic
    print(f"使用L1正则化，lambda: {args.lambda_reg}")

    # 模型初始化
    print('初始化模型...', end=' ')
    model = SNN(
        omic_input_dim=args.omic_input_dim,
        model_size_omic=args.model_size_omic,
        n_classes=args.n_classes
    )
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print('完成!')
    print_network(model)

    print('初始化优化器...', end=' ')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    print('完成!\n')

    print('设置早停机制...', end=' ')
    early_stopping = EarlyStoppingWithCIndex(
        warmup=5, patience=7, stop_epoch=20, verbose=True, 
        fold=cur, results_dir=args.results_dir, repeat=repeat
    ) if args.early_stopping else None
    print('完成!\n')

    # 训练循环
    for epoch in range(args.max_epochs):
        # 在训练集上训练一个epoch
        train_loop_survival_omic(
            epoch, model, train_loader, optimizer, args.n_classes,
            writer, loss_fn, reg_fn, args.lambda_reg, args.gc
        )
        
        # 在内部验证集上验证（用于早停）
        stop = validate_survival_omic(
            cur, epoch, model, val_loader, args.n_classes,
            early_stopping, writer, loss_fn,
            reg_fn, args.lambda_reg, args.results_dir
        )
        
        if stop:
            print(f"[训练] 在epoch {epoch} 早停")
            break
        print(f"[Epoch {epoch} 结束]\n")

    # 加载最佳模型权重 - 包含repeat编号
    if repeat is not None:
        best_cindex_ckpt = os.path.join(args.results_dir, f"repeat{repeat}_s_{cur}_best_cindex_checkpoint.pt")
        best_loss_ckpt = os.path.join(args.results_dir, f"repeat{repeat}_s_{cur}_best_loss_checkpoint.pt")
        final_model_path = os.path.join(args.results_dir, f"repeat{repeat}_s_{cur}_final_test_model.pt")
    else:
        best_cindex_ckpt = os.path.join(args.results_dir, f"s_{cur}_best_cindex_checkpoint.pt")
        best_loss_ckpt = os.path.join(args.results_dir, f"s_{cur}_best_loss_checkpoint.pt")
        final_model_path = os.path.join(args.results_dir, f"s_{cur}_final_test_model.pt")

    if os.path.exists(best_cindex_ckpt):
        model.load_state_dict(torch.load(best_cindex_ckpt))
        print(f"加载最佳C-index模型: {best_cindex_ckpt}")
    elif os.path.exists(best_loss_ckpt):
        model.load_state_dict(torch.load(best_loss_ckpt))
        print(f"⚠️ 未找到C-index最佳模型，使用loss最佳模型: {best_loss_ckpt}")
    else:
        print("⚠️ 未找到任何保存的模型，使用当前模型权重")

    # 在测试集上评估最终性能
    print("在测试集上评估最终性能...")
    test_results, test_cindex = summary_survival_omic(model, test_loader, args.n_classes)

    print(f'最终测试集 C-index: {test_cindex:.4f}')
    
    # 保存最终用于测试的模型权重（用于复现）
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_cindex': test_cindex,
        'epoch_stopped': early_stopping.best_cindex_epoch if early_stopping else args.max_epochs,
        'args': vars(args),
        'seed': args.seed,
        'repeat': repeat,
        'fold': cur
    }, final_model_path)
    print(f"保存最终测试模型权重: {final_model_path}")
    
    if args.log_data and writer is not None:
        writer.add_scalar('final/test_cindex', test_cindex, 0)
        writer.close()
    
    return test_results, test_cindex


def train_loop_survival_omic(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0.0, gc=16):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    train_loss_surv, train_loss, all_risk_scores, all_censorships, all_event_times = 0., 0., [], [], []

    for batch_idx, (x, label, time, c) in enumerate(loader):
        x = x.to(device)
        label = label.to(device)
        time = time.to(device)
        c = c.to(device)

        # 前向传播
        h = model(x_omic=x) # 假设模型输入是 x_omic

        # 计算主损失 (NLLSurvLoss)
        loss = loss_fn(h=h, y=label, t=time, c=c)
        loss_value = loss.item()

        # 计算手动正则化损失 (L1)
        loss_reg = 0.0
        if reg_fn is not None and lambda_reg > 0:
            loss_reg = reg_fn(model) * lambda_reg

        # 统计指标
        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        if label.shape[0] == 1 and (batch_idx + 1) % 100 == 0:
            print(f"batch {batch_idx}, loss: {(loss_value + loss_reg):.4f}, "
                  f"label: {label.detach().cpu().item()}, "
                  f"event_time: {time.detach().cpu().item():.4f}, "
                  f"risk: {float(risk):.4f}, bag_size: {x.size(0)}")

        # 反向传播 (对总损失进行)
        # 如果使用梯度累积，需要先将损失除以累积步数
        loss = (loss+ loss_reg) / gc
        loss.backward()

        # 参数更新 (梯度累积)
        if (batch_idx + 1) % gc == 0 or batch_idx + 1 == len(loader):
            optimizer.step()
            optimizer.zero_grad()

        # --- 计算风险评分等用于C-index ---
        if isinstance(loss_fn, NLLSurvLoss):
            with torch.no_grad():
                hazards = torch.sigmoid(h)
                survival = torch.cumprod(1 - hazards, dim=1)
                risk = -torch.sum(survival, dim=1).cpu().numpy()

        all_censorships.append(c.cpu().numpy())
        all_event_times.append(time.cpu().numpy())
        all_risk_scores.append(risk) 

    # 统计并打印训练指标
    train_loss_surv /= len(loader)
    train_loss /= len(loader)
    all_risk_scores = np.concatenate(all_risk_scores)
    all_censorships = np.concatenate(all_censorships)
    all_event_times = np.concatenate(all_event_times)

    # 计算 C-index
    # (确保 all_censorships 的布尔类型正确)
    event_observed = (1 - np.array(all_censorships)).squeeze().astype(bool) # .squeeze() if censorships are (N,1)
    c_index = concordance_index_censored(event_observed, all_event_times.squeeze(), all_risk_scores.squeeze())[0]

    print(f'Epoch: {epoch}, train_loss_surv: {train_loss_surv:.4f}, train_loss: {train_loss:.4f} train_c_index: {c_index:.4f}')

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)


def validate_survival_omic(cur, epoch, model, loader, n_classes,
                           early_stopping=None, writer=None,
                           loss_fn=None, reg_fn=None, lambda_reg=0.0, results_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = []
    all_censorships = []
    all_event_times = []

    with torch.no_grad(): # 验证时不需要梯度
        for batch_idx, (x, label, time, c) in enumerate(loader):
            x = x.to(device)
            label = label.to(device)
            time = time.to(device)
            c = c.to(device)

            h = model(x_omic=x)

            # 检查 NaN/Inf
            if torch.isnan(h).any() or torch.isinf(h).any():
                print(f"[Warning] NaN/Inf detected in model output at batch {batch_idx}, epoch {epoch}")

            # 只计算主损失
            loss = loss_fn(h=h, y=label, t=time, c=c)
            val_loss_surv += loss.item()

            # 计算手动正则化损失 (L1)
            loss_reg = 0.0
            if reg_fn is not None and lambda_reg > 0:
                loss_reg = reg_fn(model) * lambda_reg

            val_loss += loss.item() + loss_reg

            # --- 计算风险评分等用于C-index ---
            if isinstance(loss_fn, NLLSurvLoss):
                hazards = torch.sigmoid(h)
                survival = torch.cumprod(1 - hazards, dim=1)
                risk = -torch.sum(survival, dim=1).cpu().numpy()

            all_risk_scores.append(risk)
            all_censorships.append(c.cpu().numpy())
            all_event_times.append(time.cpu().numpy())

    val_loss_surv /= len(loader)
    val_loss /= len(loader)

    all_risk_scores = np.concatenate(all_risk_scores)
    all_censorships = np.concatenate(all_censorships)
    all_event_times = np.concatenate(all_event_times)

    event_observed = (1 - np.array(all_censorships)).squeeze().astype(bool)
    c_index = concordance_index_censored(event_observed, all_event_times.squeeze(), all_risk_scores.squeeze())[0]

    print(f'Val Epoch: {epoch}, val_loss_surv: {val_loss_surv:.4f}, val_loss: {val_loss:.4f}, val_c_index: {c_index:.4f}')

    # --- 早停 + 保存模型 ---
    stop_training = False
    if early_stopping:
        # 这里调用你的类，传入 val_loss 和 val_cindex
        early_stopping(epoch, val_loss, c_index, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}.")
            stop_training = True

    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

    print(f"[Validation] Epoch {epoch} finished | val_loss_surv: {val_loss_surv:.4f}, val_loss: {val_loss:.4f}, val_c_index: {c_index:.4f}, stop_training: {stop_training}")

    return stop_training

#对模型在测试集上的生存分析性能进行综合评估,回两个关键结果：
#patient_results：一个字典，包含每个样本的详细预测信息（风险评分、标签、生存时间、是否删失等）。
#c_index：整个数据集的 Concordance Index（C-index），用于量化模型的全局预测准确性。
def summary_survival_omic(model, loader, n_classes): # 不需要 reg_fn 或 lambda_reg    #校正后加device
    #校正之前
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = model.to(device)
    model.eval()

    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    patient_results = {} # 用于存储每个病人的结果

    # 获取原始数据集和索引（假设loader.dataset是Subset对象）
    try:
        original_dataset = loader.dataset.dataset  # 获取原始数据集
        indices_in_original = loader.dataset.indices  # 获取Subset使用的原始索引
    except AttributeError:
        # 如果不是Subset，直接使用原始数据集
        original_dataset = loader.dataset
        indices_in_original = list(range(len(original_dataset)))

    # 尝试获取case_ids（根据实际数据集结构调整）
    case_ids = getattr(original_dataset, 'case_ids', None)
    if case_ids is None:
        # 如果数据集没有case_ids属性，使用索引作为标识
        case_ids = indices_in_original

    for batch_idx, (x, label, time, c) in enumerate(loader):

        x = x.to(device)

        #模型预测。RNA 数据（omics 特征）通过模型得到风险预测：
        with torch.no_grad():
            h = model(x_omic=x) 

        hazards = torch.sigmoid(h)  #计算风险概率。将模型输出通过 sigmoid 函数转换为每个时间间隔的风险概率
        survival = torch.cumprod(1 - hazards, dim=1)  #计算生存函数。通过累积乘积计算每个时间点的生存概率
        risk = -torch.sum(survival, dim=1)  #计算风险分数。取负的累积生存函数之和作为最终风险分数
        risk = risk.cpu().numpy()[0] # 获取标量值
    
        # .item() 用于从单个元素的tensor中获取Python数字
        current_label = label.item() if torch.is_tensor(label) and label.numel() == 1 else label
        current_time = time.item() if torch.is_tensor(time) and time.numel() == 1 else time
        current_c = c.item() if torch.is_tensor(c) and c.numel() == 1 else c

        all_risk_scores.append(risk)
        all_censorships.append(current_c)
        all_event_times.append(current_time)

         # 确定患者标识
        if case_ids is not None:
            # 使用原始数据集中的case_id作为键
            patient_key = case_ids[indices_in_original[batch_idx]]
        else:
            # 若无case_id，使用原始索引作为键
            patient_key = indices_in_original[batch_idx]

        patient_results[patient_key] = {
            'risk': risk,
            'disc_label': current_label,
            'survival_months': current_time,
            'censorship': current_c
        }

    event_observed = (1 - np.array(all_censorships)).astype(bool)

    c_index = concordance_index_censored(event_observed, all_event_times, all_risk_scores)[0]

    return patient_results, c_index