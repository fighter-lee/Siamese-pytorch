# 自适应学习率
1.优化器
adam
Init_lr = 1e-2
Min_lr = Init_lr * 0.01
lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1
lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
Init_lr_fit = min(max(train_batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
Min_lr_fit = min(max(train_batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
