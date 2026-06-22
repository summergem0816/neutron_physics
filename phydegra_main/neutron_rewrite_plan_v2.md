# phydegra_main 当前状态核对与下一版重写计划

## 1. 当前代码版本确认

当前 `phydegra_main` 是你之前上传到服务器并真实跑过、之后又下载回本地的版本。  
从目录结构和文件内容看，它与我们此前在 `DegFlow-main` 中改过的版本基本一致，包含：

- 中子数据集读取：
  - `datasets/neutron_deg.py`
- 第一阶段训练：
  - `trainers/lit_ae.py`
- 第二阶段训练：
  - `trainers/lit_rf.py`
- 物理退化模块：
  - `models/neutron/physics_forward.py`
- 中子版配置：
  - `configs/train_neutron_lit_ae.yaml`
  - `configs/train_neutron_lit_rf.yaml`
  - `configs/models/neutron_ae.yaml`
  - `configs/models/neutron_rf.yaml`

因此，后续所有修改应以 `phydegra_main` 为准，不再以 `DegFlow-main` 为准。


## 2. 第二阶段训练日志问题定位

### 2.1 不是 `reoriganize_5.24_ae_test2_all_pair_not_group.log`

你提到的：

- `phydegra_main/reoriganize_5.24_ae_test2_all_pair_not_group.log`

不是第二阶段训练日志。这个文件只有一行：

`Processed 2500 files into result/pair_not_group_test2_all`

它只是整理 AE 推理结果的脚本输出。

### 2.2 真正的第二阶段训练日志

真正的第二阶段训练日志是：

- `phydegra_main/train_5.24_test_stage2.log`

### 2.3 报错原因

日志显示第二阶段训练本身已经跑到了：

- `Checkpoint saved at ./checkpoints/neutron_rectified_flow/rectified_flow-step20000-versionneutron_rf.pth`

随后在验证阶段报错。

错误核心是：

`RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`

发生位置在：

- `trainers/lit_rf.py`
- `validation_step()`

当前验证代码里做了两次 PSNR 计算：

1. 先用 GPU tensor 计算
2. 又把结果覆盖成 CPU tensor 计算结果

然后再把这个 CPU 上的 `psnr` 更新到 GPU 上的 `MeanMetric`，导致设备不一致报错。

也就是说：

- 第二阶段训练主体没有先爆
- 报错出在 **验证指标统计**
- 这属于当前代码中的一个明确 bug

### 2.4 这个问题后面如何处理

等真正开始改代码时，这个问题必须一起修掉。  
修法很简单，方向只有两个：

1. 验证阶段只保留 GPU 上的 `psnr` 计算
2. 或者让 metric 和 `psnr` 都统一放到 CPU

更合理的是第一种：  
**保持 metric 和输入都在同一设备，不再重复做 CPU 覆盖。**


## 3. 当前版本方法结构的真实状态

## 3.1 第一阶段当前真实逻辑

当前第一阶段不是严格意义上的“参考 `Towards a Universal Image Degradation Model via Content-Degradation Disentanglement` 现成代码”。

它实际上是：

- 基于 DegFlow 的 AE 主干
- 由我们自己写的“HQ 内容参考 + LQ latent 表示 + 物理退化监督”训练逻辑

具体是：

- `HQ -> encoder -> (z_hq, feature_hq)`
- `LQ -> encoder -> (z_t, feature_t)`
- `decoder(z_t, feature_hq) -> LQ 重建`
- `physics_forward(HQ, t, degradation_latent=z_t) -> 物理退化监督`

因此，当前第一阶段只能称为：

**弱解耦版本**

而不能称为：

**严格按照 content-degradation disentanglement 结构实现的版本**

## 3.2 当前物理退化模块的真实状态

当前物理退化模块位于：

- `models/neutron/physics_forward.py`

它目前内部只有一条统一退化链：

1. transmission
2. blur
3. scatter background
4. poisson noise
5. readout noise

并且当前所有退化的主控制量都是：

- `t`

`z_t` 只通过一个全局平均强度去轻微修正 `t`：

`effective_t = clamp(t + 0.15 * (latent_strength - 0.5), 0, 1)`

所以当前版本并没有实现你现在希望的那种：

- 固定物理层
- `t` 主控统计层
- `z_t` 辅助控制部分统计层


## 4. 下一版的重写目标

下一版重写要做两件事，而且这两件事要分开处理：

### A. 重写物理退化模块

目标：

- 按照
  - `g = P(S[f] * h_geo * h_det + b_scatter)`
  的结构重写
- 把 Geant4 的固定几何参数显式引入
- 把物理层拆开，而不是继续用一个统一 `effective_t`

### B. 重写内容-退化模块

目标：

- 参考 `Towards a Universal Image Degradation Model via Content-Degradation Disentanglement`
- 将当前“共享 encoder 的弱解耦版本”
- 改成更接近论文结构的：
  - homogeneous/global degradation branch
  - inhomogeneous/local degradation branch
  - degradation-aware synthesis branch


## 5. 下一版物理退化模块的建议框架

## 5.1 目标公式

下一版建议直接按下面的成像链组织：

`g = P((S[f] * h_geo * h_det) + b_scatter) + n_read`

其中：

- `f`：内容参考图像或 clean/content estimate
- `S`：源强/通量层
- `h_geo`：几何模糊层
- `h_det`：探测器系统模糊层
- `b_scatter`：散射背景层
- `P`：Poisson 统计层
- `n_read`：读出噪声层

## 5.2 退化层分工

### 固定物理层

这些层不应由粒子数主导：

- `h_geo`
- `h_det`

这两层主要由 Geant4 参数决定，只允许：

- 用 Geant4 参数初始化
- 在训练中微调参数

但不建议：

- 由 `z_t` 直接控制

### 可变统计层

这些层受粒子数变化影响明显：

- `S`
- `b_scatter`
- `P`
- `n_read`

这些层建议采用：

- `t` 主控
- `z_t` 辅助修正

即：

- `t` 决定目标退化等级的大趋势
- `z_t` 负责样本级微调


## 6. Geant4 中可直接进入物理模块的初始参数

根据当前未注释版本 Geant4 代码，可直接提取：

### 探测器像元与阵列

- 阵列尺寸：`338 x 338`
- 像元有效宽度：`0.6 mm`
- pitch：`0.66 mm`
- 闪烁体厚度：`40 mm`

### 束流参数

- 粒子类型：neutron
- 粒子能量：`2.5 MeV`
- 角发散：`1 deg`
- 束斑半径：`16.5 cm`

### 相对位置

- 物体中心约在 `z = 50 mm`
- 探测器中心约在 `z = 0`
- 因此物体到探测器距离第一版可近似取 `L_od ≈ 50 mm`

### 这些参数在下一版中的角色

#### 作为 `h_geo` 初始参数

- `theta_div = 1 deg`
- `L_od ≈ 50 mm`
- `pitch = 0.66 mm`

用于初始化几何模糊核宽度。

#### 作为 `h_det` 初始参数

- `pixel width = 0.6 mm`
- `pitch = 0.66 mm`
- `scintillator thickness = 40 mm`

用于初始化探测器模糊核宽度。

需要明确：

- 这些参数不能保证绝对准确
- 更适合作为**可学习参数的初始值**


## 7. 下一版内容-退化模块的建议重写方向

## 7.1 参考论文的关键结构

从论文 `Towards a Universal Image Degradation Model via Content-Degradation Disentanglement` 的方法部分可以抽出三个关键结构：

1. HDEN：Homogeneous Degradation Encoding Network
2. IDEN：Inhomogeneous Degradation Encoding Network
3. IDA-SFT based degradation synthesis network

其思想核心是：

- 将全局退化和局部退化分开编码
- 再将退化表示注入到退化生成网络中

## 7.2 对我们的任务如何映射

中子成像任务里，可以做如下映射：

### 内容分支

负责提取：

- 物体结构
- 边缘
- 透射分布

### 全局退化分支

对应：

- 粒子统计强弱
- 全局噪声水平
- 全局对比退化

### 局部退化分支

对应：

- 局部散射背景差异
- 局部非均匀退化
- 某些区域的结构可见性下降

### 退化生成/注入模块

负责把：

- 内容表示
- 全局退化表示
- 局部退化表示

共同送入下一版物理退化模块

## 7.3 下一版不是照搬论文，而是“参考结构重写”

这里必须明确：

- 我们本地只有论文 PDF，没有该论文的原始代码仓库内容
- 因此不能说“直接搬现成代码”
- 但可以按论文方法结构，在 `phydegra_main` 中重写出对应模块

也就是说，下一版会是：

**参考该论文结构的任务定制重写版本**

而不是：

**直接复制该论文原始实现**


## 8. 推荐的实际修改顺序

由于第二阶段还在跑，当前先不改代码。  
真正开始修改时，建议按下面顺序执行：

### 第一步：先修当前第二阶段验证 bug

原因：

- 不修这个，后面你继续跑第二阶段会反复在验证阶段中断

### 第二步：重写物理退化模块

先做：

- `h_geo`
- `h_det`
- `S`
- `b_scatter`
- `P`
- `n_read`

并把 Geant4 参数引进去

### 第三步：重写内容-退化模块

把当前共享 encoder 的弱解耦，改成：

- 内容分支
- homogeneous/global degradation 分支
- inhomogeneous/local degradation 分支
- 退化注入模块

### 第四步：重接第一阶段训练逻辑

确保：

- 新内容-退化模块
- 新物理退化模块

能在第一阶段一起工作

### 第五步：再适配第二阶段 flow

确保第二阶段输出的 latent 或退化表示，能正确进入新物理退化模块


## 9. 当前阶段结论

当前最合理的推进策略是：

1. 先确认并固定下一版结构设计
2. 优先重写物理退化模块
3. 再重写内容-退化结构
4. 修改时全部在 `phydegra_main` 上进行

当前不建议立即改代码的原因是：

- 第二阶段训练还未结束
- 新物理模块和新内容-退化结构需要先统一设计接口

因此，后续应先按本文件确认方案，再逐步落代码。

