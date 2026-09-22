# 两系统 RKVAE 实验入口

使用安装了 torch、scipy、gym、matplotlib、pybullet 的 DL 环境，在仓库根目录运行。代码只注入状态转移后的单个过程噪声 v；每 epoch 完整遍历数据，验证和早停使用 epoch 语义；不使用控制约束损失。

```bash
/home/xyrrrrrrrr/.conda/envs/DL/bin/python experiments/check_rkvae.py
/home/xyrrrrrrrr/.conda/envs/DL/bin/python experiments/rkvae.py train --system damping --epochs 100 --seed 1 --device cuda:0 --output results/paper/damping/seed1/train
/home/xyrrrrrrrr/.conda/envs/DL/bin/python experiments/rkvae.py evaluate --checkpoint results/paper/damping/seed1/train/best.pt --mode lifted --eta 0.1 --sigma 0.1 --trials 100 --horizon 200 --output results/paper/damping/seed1/lifted_sigma01
```

Franka 用 `--system franka` 训练，评估用 `--task eight` 或 `--task star --horizon 3000 --trials 20`。训练变体 `--variant joint|deterministic|no_kl|bilinear|lifted`；模式 `--mode bilinear|lifted`；零输入对照 `--controller zero`。确定性消融保留模型容器但不用方差采样或 KL，方差头不参与有效训练。

矩阵入口默认只打印命令，不启动任务：

当前训练噪声默认为 `0.002`，评估噪声默认为 `[0, 0.0005, 0.001, 0.002]`。
lifted 和 bilinear 共用 A，各自预测真实下一步状态编码，loss 沿用逐时间步 MSE 求和。
修改损失后需重新训练；`run_e2.py` 仅评估已有 checkpoint，而 `run_matrix.py --execute`
会先训练再评估。单种子 E2 pendulum 可使用
`python experiments/run_matrix.py --systems damping --seeds 2 --root results/e2_pendulum_lifted_consistency --execute`。

```bash
/home/xyrrrrrrrr/.conda/envs/DL/bin/python experiments/run_matrix.py --device cuda:0
# 添加 --execute 顺序执行两个系统、5 个种子、两模式、4 档噪声的主实验。
```

矩阵默认 eta=0，用于无残差反馈基准；正式完整方法应在验证集选定 eta 后传给评估入口。矩阵不自动调参，也不把零残差反馈结果标成完整控制器。

## 输出和实验定义

训练保存 config.json、history.jsonl、best.pt、last.pt。不同运行必须使用不同目录。评估保存 metrics.json、逐轨迹 trial_XXX.npz 和 control.png。NPZ 包含真实状态、物理输入、提升输入、参考、过程噪声、残差、控制耗时；Franka 另含实际末端位置和目标位置。失败轨迹保留；完成比例与完成轨迹成本分别报告。成本使用实际物理输入及终端状态成本。

DampingPendulum 使用 SinglePendulum，Q=diag(5,0.01)、R=0.1，物理输入限幅为环境的 ±8。测试交替使用局部与较大偏差初始状态。Franka 使用14维独立关节位置/速度状态，物理控制为7维速度命令，末端坐标由正向运动学重算。Franka 用 IK 参考关节和状态反馈进行 Eight/Star 跟踪；它是本实验的明确实现选择，不是任意参考跟踪稳定性的证明。

缩放是预先固定的物理尺度，不拟合测试数据：摆为 [pi,8]；Franka 关节位置为1 rad、速度为2.175 rad/s，以默认姿态为中心。sigma 定义在这些归一化坐标下。与计划中训练集拟合缩放相比，这是为保证所有方法共享尺度的替代方案，应在正式论文中明确。不得复用旧17维 Franka 或旧观测噪声模型 checkpoint；入口严格加载新 checkpoint。

## 已验证与待接入

已用小规模数据执行两个系统训练、DampingPendulum 两种控制模式及非零 eta、Franka lifted 跟踪，均完成 rollout 并生成图像；check_rkvae.py 验证梯度、Kronecker 排列、控制逆变换、噪声写入真实状态和数据种子复现。

此入口覆盖 RKVAE 主实验和模块消融。实验计划中的 DKUC/DKAC/FLC/LS/KRBF 公平基线适配、独立预测测试集与 E5 有限样本/比例界诊断尚未实现，不能用本入口的 RKVAE 结果代替这些证据。延迟目前包括第一步（无预热剔除），用于调试；正式 E4 需增加预热和固定硬件测量协议。

PyBullet 当前 URDF 会警告缺少惯性参数；这属于现有仿真模型。机器人论文实验前需审计该模型的物理可信度，成功运行不代表已完成该审计。

## 多种子汇总

```bash
/home/xyrrrrrrrr/.conda/envs/DL/bin/python experiments/summarize.py results/rkvae_paper --output results/rkvae_paper/summary.json
```

先汇总每个训练种子的轨迹，再对独立训练种子计算标准差和 bootstrap 95% 区间。单种子时不生成伪置信区间；重复训练种子会报错。失败率保留，成本统计只含完成轨迹。

## 独立控制验证与 checkpoint 选择

`best.pt` 保存所有已验证轮次中的最低预测损失，不再受 `min_delta` 限制；
`min_delta` 仅用于判断早停的显著改善。每次预测验证还保存 `candidates/epoch_XXXX.pt`。

pendulum 矩阵默认训练 200 轮，早停耐心 200 轮。训练后自动调用
`select_control_checkpoint.py`：候选为每 10 轮 checkpoint 加 `best.pt` 和 `last.pt`，
用独立种子 `2000001..2000020`、每档 20 条轨迹、200 步、四档噪声
`[0, 0.0005, 0.001, 0.002]` 验证。所有候选使用相同验证初值和噪声。
按完成率、成功率、负平均代价依次排序，同分保留较早轮次；两种模式分别选择
`control_validation/deploy_lifted.pt` 与 `deploy_bilinear.pt`。
`selection.json` 和原始验证轨迹保留供审计。此选择器目前仅支持 pendulum。
最终评估仍用种子 `10001..10100`，不参与选择。已有旧实验只有 best/last 时，
选择器仅比较这两个模型，无法恢复被覆盖的历史 checkpoint。
