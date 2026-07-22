# phydegra_main 下一版详细重写方案

## 1. 目标

下一版需要同时完成三件事：

1. 重写物理退化模块  
   按照
   `g = P(S[f] * h_geo * h_det + b_scatter) + n_read`
   的结构改造，并把 Geant4 参数显式引入。

2. 重写内容-退化模块  
   参考 `Towards a Universal Image Degradation Model via Content-Degradation Disentanglement` 的结构思想，将当前“共享 encoder 的弱解耦版本”改成显式的内容分支、全局退化分支和局部退化分支。

3. 修复第二阶段验证阶段的 device bug  
   等结构方案落地并改完代码后，一并修复 `trainers/lit_rf.py` 验证阶段 CPU/GPU 混用的问题。


## 2. 总体思路

下一版仍然保留“两阶段训练”的大框架，不推翻现在已经跑通过的流程。

### 第一阶段

输入：

- `I_HQ`
- `I_t`
- `t`

学习：

- 内容表示
- 退化表示
- 物理退化生成

### 第二阶段

输入：

- 同一组样本的多粒子数图像

学习：

- 紧凑退化状态在潜空间中的连续演化

最终推理：

- `HQ -> 内容分支 -> clean/content reference`
- `HQ -> 退化分支 -> z_0`
- `(z_0, t) -> latent flow -> \hat{z}_t`
- `(clean/content reference, t, \hat{z}_t) -> 新物理退化模块 -> \hat{LQ}`


## 3. 下一版内容-退化模块设计

## 3.1 当前问题

当前版本中：

- `HQ` 和 `LQ` 复用同一个 encoder
- `feature_hq` 只进 decoder
- `z_t` 只作为弱退化表示

这只能算“弱解耦”，不是显式内容-退化分离。

## 3.2 下一版目标结构

下一版建议显式拆成四部分：

1. 内容编码器 `E_c`
2. 全局退化编码器 `E_g`
3. 局部退化编码器 `E_l`
4. 退化压缩器 `C_d`

### 3.2.1 内容编码器 `E_c`

输入：

- `I_HQ`

输出：

- `z_c`
- 多尺度内容特征 `H_c`
- 可选 clean reference `f_clean`

作用：

- 提取物体几何结构
- 提取材料透射结构
- 为后续物理退化模块提供更“干净”的内容参考

建议实现：

- 保留当前 `AE encoder + multi-scale skip feature` 的思路
- 但将其明确命名为内容分支

### 3.2.2 全局退化编码器 `E_g`

参考论文中的 HDEN（Homogeneous Degradation Encoding Network）。

输入：

- `I_t`

输出：

- 全局退化向量 `e_g`

作用：

- 表达全局统计噪声水平
- 表达全局对比退化
- 表达全局退化强度

建议结构：

- 双分支 CNN
- 一个短程分支提取局部统计
- 一个长程分支提取全局感受野信息
- 全局池化后接 MLP 输出向量

建议输出形状：

- `e_g : [B, Cg]`

推荐 `Cg = 128` 或 `256`

### 3.2.3 局部退化编码器 `E_l`

参考论文中的 IDEN（Inhomogeneous Degradation Encoding Network）。

输入：

- `I_t`

输出：

- 局部退化图 `e_l`

作用：

- 表达空间变化的散射背景
- 表达局部不均匀退化
- 表达局部细节可见性差异

建议结构：

- 类似 U-Net 编码器
- 保留空间结构
- 输出低分辨率局部退化 map

建议输出形状：

- `e_l : [B, Cl, H/8, W/8]`

推荐 `Cl = 32` 或 `64`

### 3.2.4 退化压缩器 `C_d`

这是为了兼容第二阶段 latent flow 的关键模块。

输入：

- `e_g`
- `e_l`

输出：

- 紧凑退化 latent `z_t`

作用：

- 将显式的全局/局部退化表示压缩为第二阶段可学习的紧凑退化状态

建议结构：

- 将 `e_g` broadcast 后与 `e_l` 拼接
- 经过若干卷积/残差块
- 输出与 flow 网络兼容的空间 latent

建议输出形状：

- `z_t : [B, Cz, H/8, W/8]`

其中 `Cz` 保持与当前 flow 兼容，建议：

- `Cz = 4`

这样可以最大限度复用当前第二阶段 NCSN++ 接口。


## 4. 第二阶段如何与新结构兼容

## 4.1 第二阶段不再学习“内容 latent 轨迹”

下一版建议明确：

- 第二阶段只学习退化状态轨迹
- 不学习内容轨迹

也就是说，第二阶段 flow 的目标不是：

- `content latent -> degraded image latent`

而是：

- `degradation state at HQ -> degradation state at target t`

## 4.2 新的第二阶段输入输出

第一阶段中，对同一组样本：

- `I_HQ -> E_g/E_l/C_d -> z_0`
- `I_50 -> E_g/E_l/C_d -> z_50`
- `I_30 -> E_g/E_l/C_d -> z_30`
- `I_20 -> E_g/E_l/C_d -> z_20`
- `I_10 -> E_g/E_l/C_d -> z_10`

第二阶段学习轨迹：

- `{z_0, z_50, z_30, z_20, z_10}`

推理时：

- `HQ -> 内容分支 -> f_clean`
- `HQ -> 退化分支 -> z_0`
- `(z_0, t) -> flow -> \hat{z}_t`
- `physics_forward(f_clean, t, \hat{z}_t) -> \hat{LQ}`

这样有两个优点：

1. 与当前第二阶段 DegFlow 风格兼容
2. 明确把内容与退化状态分开


## 5. 下一版物理退化模块设计

## 5.1 总体形式

建议写成：

`g = P((S[f] * h_geo * h_det) + b_scatter) + n_read`

分六层实现：

1. 源强/通量层 `S`
2. 几何模糊层 `h_geo`
3. 探测器模糊层 `h_det`
4. 散射背景层 `b_scatter`
5. Poisson 统计层 `P`
6. 读出噪声层 `n_read`

## 5.2 设计原则

### 固定层

这些层由 Geant4 参数初始化，允许训练微调，但不由 `z_t` 控制：

- `h_geo`
- `h_det`

### 可变层

这些层由 `t` 主控、`z_t` 辅助：

- `S`
- `b_scatter`
- `P`
- `n_read`


## 6. Geant4 参数如何进入物理退化模块

根据当前仿真代码，可提取的高参考性参数：

- `theta_div = 1 deg`
- `pixel width = 0.6 mm`
- `pitch = 0.66 mm`
- `scintillator thickness = 40 mm`
- `L_od ≈ 50 mm`
- `particle levels = {3e8, 5e7, 3e7, 2e7, 1e7}`

这些参数不保证绝对准确，但适合作为可学习参数的初值。


## 7. 每一层的建议构造方式

## 7.1 通量层 `S`

输入：

- `f_clean`
- `t`
- `z_t`

输出：

- `x_s`

建议形式：

- `t` 决定等效粒子通量的大趋势
- `z_t` 给出小幅修正

建议公式：

- `N_base(t)` 由固定粒子数映射插值得到
- `delta_N(z_t)` 由小型 MLP 或 pooled conv head 得到
- `N_eff = N_base(t) * exp(delta_N(z_t))`
- `x_s = N_eff * f_clean`

说明：

- 如果希望 `P` 那一层再乘粒子数，也可以将 `S` 理解为 gain 层
- 但从实现上，统一由 `N_eff` 控制更直接

## 7.2 几何模糊层 `h_geo`

输入：

- `x_s`

输出：

- `x_geo`

控制方式：

- 不使用 `t`
- 不使用 `z_t`
- 仅使用可学习的固定参数 `sigma_geo`

初始化方式：

- `sigma_geo_mm_init = L_od * tan(theta_div)`
- 由毫米换算成像素：
  - `sigma_geo_px_init = sigma_geo_mm_init / pitch`

按当前参数：

- `L_od ≈ 50 mm`
- `theta_div = 1 deg`
- `pitch = 0.66 mm`

则：

- `sigma_geo_mm_init ≈ 50 * tan(1 deg) ≈ 0.873 mm`
- `sigma_geo_px_init ≈ 0.873 / 0.66 ≈ 1.32 px`

建议实现：

- 用 `nn.Parameter` 存储一个可学习的 `log_sigma_geo`
- 通过 `softplus` 保证其为正

## 7.3 探测器模糊层 `h_det`

输入：

- `x_geo`

输出：

- `x_det`

控制方式：

- 不使用 `t`
- 不使用 `z_t`
- 仅使用固定可学习参数 `sigma_det`

初始化思路：

像元积分 blur 可由 box blur 的高斯等效近似：

- `sigma_pixel_mm = w_pixel / sqrt(12)`

其中：

- `w_pixel = 0.6 mm`

则：

- `sigma_pixel_mm ≈ 0.173 mm`
- `sigma_pixel_px ≈ 0.173 / 0.66 ≈ 0.26 px`

然后引入一个由闪烁体厚度导致的附加固定模糊项：

- `sigma_scint_extra_px`

第一版可初始化为：

- `0.2 ~ 0.4 px`

于是：

- `sigma_det_init = sqrt(sigma_pixel_px^2 + sigma_scint_extra_px^2)`

建议最终初始化在：

- `0.35 ~ 0.5 px`

## 7.4 散射背景层 `b_scatter`

输入：

- `x_det`
- `t`
- `z_t`
- 可选 `e_l`

输出：

- `x_scatter`

控制方式：

- `t` 主控
- `z_t` 辅助
- 如果引入 `e_l`，则用它表达局部不均匀背景

建议形式分为两部分：

### 全局散射项

- `beta_g = beta_0 + beta_t(t) + delta_beta_g(z_t)`

### 局部散射项

- `M_scatter = Conv(e_l)` 或 `Conv(z_t)` 得到归一化 map
- `beta_l = delta_beta_l(z_t)`

最后：

- `b_scatter = beta_g * Blur_large(x_det) + beta_l * M_scatter`

说明：

- 如果第一版想保守一点，局部散射项可以暂时先不用 `e_l`
- 先只做“全局低频背景 + z_t 小幅修正”

## 7.5 Poisson 统计层 `P`

输入：

- `x_det + b_scatter`
- `N_eff`

输出：

- `x_poisson`

控制方式：

- `t` 主控
- `z_t` 通过 `N_eff` 已经间接参与

建议实现：

- `x_poisson = Poisson(clamp(x_in, 0, +inf)) / N_eff`

说明：

- 这一层是最物理的一层
- 应尽量保留

## 7.6 读出噪声层 `n_read`

输入：

- `x_poisson`
- `t`
- `z_t`

输出：

- `g`

控制方式：

- `t` 主控
- `z_t` 辅助

建议形式：

- `sigma_read = sigma_read_base(t) * exp(delta_read(z_t))`
- `g = x_poisson + Normal(0, sigma_read)`


## 8. 下一版物理模块推荐代码结构

建议将当前单一 `physics_forward.py` 重构为：

### 8.1 模块拆分

- `GeoBlurLayer`
- `DetectorBlurLayer`
- `ScatterLayer`
- `PoissonLayer`
- `ReadoutNoiseLayer`
- `NeutronPhysicalForwardV2`

### 8.2 控制头

增加两个小控制头：

- `FluxControlHead(z_t, t) -> delta_N`
- `NoiseScatterControlHead(z_t, t) -> delta_scatter, delta_read`

如果后续引入 `e_l`：

- `LocalScatterHead(e_l) -> local_scatter_map`

### 8.3 参数管理

固定物理参数不要写死在 `forward` 里，建议都写到 `__init__` 里作为：

- 初值
- 是否可学习
- 上下界约束


## 9. 第一阶段建议训练方式

下一版第一阶段不建议一步就完全照抄论文全部训练细节，而是采用“结构优先、训练稳妥”的方式。

建议第一阶段训练：

### 9.1 内容分支训练目标

- 从 `HQ` 重建 `HQ` 或输出 `clean reference`

建议损失：

- `L_clean = ||f_clean - HQ||`

### 9.2 退化分支训练目标

- `I_t -> E_g/E_l/C_d -> z_t`

作用：

- 学到可压缩的退化状态表示

### 9.3 物理分支训练目标

- `g_hat = Physics(f_clean, t, z_t, e_l)`

与真实 `LQ` 对齐：

- `L_phys = ||g_hat - I_t||`

### 9.4 可选辅助重建支路

为了训练稳定，建议保留一个辅助图像重建头：

- `L_aux = ||AuxDecoder(z_t, H_c) - I_t||`

这个支路的作用是：

- 帮助内容分支与退化分支快速对齐
- 避免一上来完全依赖物理退化模块导致难训

建议把它定义为：

- 训练辅助支路
- 推理时不作为最终输出

### 9.5 第一阶段建议总损失

建议第一版总损失：

- `L_stage1 = lambda_clean * L_clean + lambda_phys * L_phys + lambda_aux * L_aux`

推荐初值：

- `lambda_clean = 1.0`
- `lambda_phys = 1.0`
- `lambda_aux = 0.5`

说明：

- 不建议第一版就引入论文中的完整 entropy regularization
- 因为当前没有现成官方代码可直接对照，且加入后训练稳定性风险较大
- 第一版先完成“显式结构分离 + 物理层重写”


## 10. 第二阶段建议训练方式

第二阶段保持 DegFlow 主体不变，只替换第一阶段提供给它的退化 latent。

### 当前第二阶段

- 轨迹点来自共享 encoder latent

### 下一版第二阶段

- 轨迹点来自 `C_d(E_g(I_t), E_l(I_t))`

即：

- 用新的紧凑退化 latent `z_t` 训练 flow

### 第二阶段损失

仍建议保留：

- `L_flow`
- `L_lpips`
- `L_phys`

只是 `L_phys` 中调用的是新版物理模块。


## 11. 第二阶段验证 bug 的修改要求

当前 `trainers/lit_rf.py` 在验证阶段有 bug：

- 先在 GPU 上算一次 `psnr`
- 又覆盖成 CPU 上算的 `psnr`
- 再更新 GPU 上的 metric

因此后续代码修改完成后，必须顺手修复：

### 建议修法

- 删除 CPU 覆盖计算
- metric 与 `psnr` 保持同设备
- 统一在 GPU 上更新

这一项属于必须一起修复的收尾 bug。


## 12. 最终建议的修改顺序

### 第一步

先重写：

- 内容编码器 `E_c`
- 全局退化编码器 `E_g`
- 局部退化编码器 `E_l`
- 退化压缩器 `C_d`

### 第二步

重写新版物理退化模块：

- `S`
- `h_geo`
- `h_det`
- `b_scatter`
- `P`
- `n_read`

### 第三步

改第一阶段训练器：

- 接新内容-退化结构
- 接新版物理模块

### 第四步

改第二阶段训练器：

- 用新 `z_t` 轨迹替换当前 latent 轨迹
- 保持 flow 主体框架不变

### 第五步

修复第二阶段验证指标 device bug

### 第六步

如有新增依赖，再补写 `requirements.txt`


## 13. 当前阶段结论

下一版最合理的工程路线是：

- 不推翻现有两阶段结构
- 保留第二阶段 latent flow 的主框架
- 将第一阶段显式改造成：
  - 内容分支
  - 全局退化分支
  - 局部退化分支
  - 退化压缩器
  - 基于物理层的退化生成模块

同时将物理退化明确拆成：

- 固定几何/探测器层
- `t` 主控、`z_t` 辅助的统计退化层

这是当前最稳妥、最符合你目标、也最适合继续落代码的方案。

