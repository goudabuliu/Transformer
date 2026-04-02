import config
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tools.data_loader import MTDataset
from model.tf_model import make_model
import logging
import sacrebleu
from tqdm import tqdm

from beam_decoder import beam_search
from model.train_utils import get_std_opt
from tools.tokenizer_utils import chinese_tokenizer_load
from tools.create_exp_folder import create_exp_folder


logging.basicConfig(format='%(asctime)s-%(name)s-%(levelname)s-%(message)s', level=logging.INFO)

# 开启 TF32，RTX 30/40/5090 专用，速度×2，显存不变
torch.set_float32_matmul_precision('medium')

def run_epoch(data, model, criterion, optimizer=None, accum_steps=1, scaler=None, use_amp=False):
    """标准训练/验证循环，支持梯度累积和AMP

    参数：
        data: DataLoader迭代器
        model: Transformer模型（已包装为DataParallel）
        criterion: 损失函数
        optimizer: 优化器（训练时提供，验证时为None）
        accum_steps: 梯度累积步数
        scaler: GradScaler实例（AMP用）
        use_amp: 是否使用混合精度训练

    返回：
        average_loss: 平均损失（每个token的损失）
    """
    total_tokens = 0.
    total_loss = 0.

    # 根据是否训练设置模型模式
    if optimizer is not None:
        model.train()
    else:
        model.eval()

    for i, batch in enumerate(tqdm(data)):
        # 将数据移动到设备
        src = batch.src.to(config.device)
        trg = batch.trg.to(config.device)
        src_mask = batch.src_mask.to(config.device)
        trg_mask = batch.trg_mask.to(config.device)
        trg_y = batch.trg_y.to(config.device)
        ntokens = batch.ntokens

        # 前向传播
        if use_amp and scaler is not None:
            with torch.amp.autocast(device_type='cuda'):
                out = model(src, trg, src_mask, trg_mask)
                # 使用模型的generator生成预测
                gen_out = model.module.generator(out)
                loss = criterion(
                    gen_out.contiguous().view(-1, gen_out.size(-1)),
                    trg_y.contiguous().view(-1)
                ) / ntokens
        else:
            out = model(src, trg, src_mask, trg_mask)
            gen_out = model.module.generator(out)
            loss = criterion(
                gen_out.contiguous().view(-1, gen_out.size(-1)),
                trg_y.contiguous().view(-1)
            ) / ntokens

        # 训练阶段：反向传播和梯度累积
        if optimizer is not None:
            # 梯度累积：损失除以累积步数
            if use_amp and scaler is not None:
                scaler.scale(loss / accum_steps).backward()
            else:
                (loss / accum_steps).backward()

            # 累积足够步数后更新参数
            if (i + 1) % accum_steps == 0 or (i + 1) == len(data):
                if use_amp and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item()
        total_tokens += ntokens

    return total_loss / total_tokens if total_tokens > 0 else 0.0

def train(train_data, dev_data, model, model_par, criterion, optimizer):
    """训练并保存模型"""
    import os  # 用于文件操作

    # best_bleu_score初始化
    best_bleu_score = -float('inf')  # 初始最佳BLEU分数为负无穷
    # 创建保存权重的路径
    exp_folder, weights_folder = create_exp_folder()

    # 创建GradScaler（如果启用AMP）
    scaler = torch.cuda.amp.GradScaler() if config.use_amp else None

    # 调试：打印优化器信息
    logging.info(f"优化器: {type(optimizer).__name__}")
    if hasattr(optimizer, 'optimizer'):
        logging.info(f"内部优化器: {type(optimizer.optimizer).__name__}")
    if hasattr(optimizer, 'rate'):
        logging.info(f"初始学习率: {optimizer.rate(1):.2e}")
    elif hasattr(optimizer, 'param_groups'):
        logging.info(f"优化器学习率: {optimizer.param_groups[0]['lr']}")

    # 开始训练循环，迭代每个epoch
    for epoch in range(1, config.epoch_num + 1):
        logging.info(f"第{epoch}轮模型训练与验证")

        # ==================== 训练阶段 ====================
        model.train()
        train_loss = run_epoch(
            train_data,
            model_par,
            criterion,
            optimizer=optimizer,
            accum_steps=config.accum_steps,
            scaler=scaler,
            use_amp=config.use_amp
        )

        # ==================== 验证阶段（每个epoch都验证）====================
        model.eval()
        dev_loss = run_epoch(
            dev_data,
            model_par,
            criterion,
            optimizer=None,  # 验证时不更新参数
            accum_steps=1,   # 验证时不使用梯度累积
            scaler=None,     # 验证时不使用AMP
            use_amp=False
        )

        # 计算模型在验证集（dev_data）上的BLEU分数
        bleu_score = evaluate(dev_data, model)
        logging.info(f"Epoch: {epoch}, train_loss: {train_loss:.3f}, val_loss: {dev_loss:.3f}, Bleu Score: {bleu_score:.2f}\n")

        # 如果当前epoch的模型的BLEU分数更优，则保存最佳模型
        if bleu_score > best_bleu_score:
            # 如果之前已存在最优模型，先删除
            if best_bleu_score != -float('inf'):
                old_model_path = f"{weights_folder}/best_bleu_{best_bleu_score:.2f}.pth"
                if os.path.exists(old_model_path):
                    os.remove(old_model_path)

            model_path_best = f"{weights_folder}/best_bleu_{bleu_score:.2f}.pth"
            # 保存当前模型的状态字典到指定路径
            torch.save(model.state_dict(), model_path_best)
            # 更新最佳BLEU分数
            best_bleu_score = bleu_score
            # 记录最佳模型保存信息到日志
            logging.info(f"保存最佳模型: {model_path_best}")

        # 保存当前模型（最后一次训练）
        if epoch == config.epoch_num:  # 判断是否达到设定的训练轮数
            model_path_last = f"{weights_folder}/last_bleu_{bleu_score:.2f}.pth"  # 构建模型保存路径，包含BLEU分数
            torch.save(model.state_dict(), model_path_last)  # 保存模型的状态字典
            logging.info(f"保存最终模型: {model_path_last}")


def evaluate(data, model):
    sp_chn = chinese_tokenizer_load()
    trg = []
    res = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(data):
            src = batch.src.to(config.device)
            src_mask = (src != 0).unsqueeze(-2).to(config.device)
            cn_sent = batch.trg_text

            # ==============================================
            # 贪婪解码，一步到位，比 beam_search 快几十倍
            # ==============================================
            batch_size = src.size(0)
            max_len = config.max_len

            # Encoder 只算一遍！！！（beam_search 里重复算 encoder 巨慢）
            memory = model.encode(src, src_mask)

            # 初始化 decoder 输入：以 <bos> 开头
            ys = torch.full((batch_size, 1), config.bos_idx, dtype=torch.long, device=config.device)

            for _ in range(max_len):
                tgt_mask = (ys != config.padding_idx).unsqueeze(-2) & \
                           SubsequentMask(ys.size(1)).to(config.device)

                out = model.decode(memory, src_mask, ys, tgt_mask)
                prob = F.log_softmax(model.generator(out[:, -1:]), dim=-1)  # 只取最后一个 token
                _, next_word = torch.max(prob, dim=-1)

                ys = torch.cat([ys, next_word], dim=1)

                # 全部句子都遇到 eos 就提前结束
                if (next_word == config.eos_idx).all():
                    break

            # 批量转句子
            for seq in ys:
                seq = seq.tolist()
                if config.eos_idx in seq:
                    seq = seq[:seq.index(config.eos_idx)]
                res.append(sp_chn.decode_ids(seq))
            trg.extend(cn_sent)

    trg = [trg]
    bleu = sacrebleu.corpus_bleu(res, trg, tokenize='zh')
    return float(bleu.score)

# 辅助函数：subsequent mask（防止看到未来 token）
def SubsequentMask(size):
    mask = (torch.triu(torch.ones((1, size, size))) == 1).transpose(1, 2)
    return mask

def test(data, model, criterion):
    with torch.no_grad():
        # 加载模型
        model.load_state_dict(torch.load(config.model_path))
        model_par = torch.nn.DataParallel(model)
        model.eval()
        # 开始预测
        test_loss = run_epoch(data, model_par, criterion, optimizer=None, accum_steps=1, scaler=None, use_amp=False)
        bleu_score = evaluate(data, model)
        logging.info('Test loss: {},  Bleu Score: {}'.format(test_loss, bleu_score))


def run():
    # 创建训练数据集和开发数据集
    # 使用MTDataset类分别加载训练数据和开发数据
    train_dataset = MTDataset(config.train_data_path)   # 初始化训练数据集，使用配置中指定的训练数据路径
    dev_dataset = MTDataset(config.dev_data_path)   # 初始化开发数据集，使用配置中指定的开发数据路径
    test_dataset = MTDataset(config.test_data_path)

    # 创建训练数据加载器，用于训练过程中批量加载数据
    # shuffle=True 表示在每个epoch开始时会打乱数据顺序，以增加模型的泛化能力
    # batch_size=config.batch_size 表示每个批次的样本数量，具体值由配置文件决定
    # collate_fn=train_dataset.collate_fn 表示自定义的数据整理函数，用于处理每个批次的数据
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size,
                                  collate_fn=train_dataset.collate_fn,
                                  num_workers=config.num_workers,
                                  pin_memory=config.pin_memory,
                                  prefetch_factor=2 if config.num_workers > 0 else None)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config.batch_size,
                                collate_fn=dev_dataset.collate_fn,
                                num_workers=config.num_workers,
                                pin_memory=config.pin_memory,
                                prefetch_factor=2 if config.num_workers > 0 else None)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size,
                                 collate_fn=test_dataset.collate_fn,
                                 num_workers=config.num_workers,
                                 pin_memory=config.pin_memory,
                                 prefetch_factor=2 if config.num_workers > 0 else None)

    # 初始化模型
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)

    #  将模型包装成数据并行模式,这样可以在多个GPU上并行处理数据，提高训练效率
    model_par = torch.nn.DataParallel(model, device_ids=config.device_id) 

    # 训练阶段，选择损失函数和优化器
    # CrossEntropyLoss是常见的分类问题损失函数，ignore_index=0表示忽略填充部分
    # reduction='sum'表示计算损失时会对所有token的损失求和
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

    # 调用get_std_opt函数获取标准的Noam优化器，这通常包括学习率调度器（如预热后衰减）
    optimizer = get_std_opt(model)

    # 开始训练
    train(train_dataloader, dev_dataloader, model, model_par, criterion, optimizer)
    # test(test_dataloader, model, criterion)


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id 
    import warnings
    warnings.filterwarnings('ignore')
    run()