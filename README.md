# Hemispheric_Fingerprints：基于深度神经网络与积分梯度特征归因方法的指纹提取工具包

> **版本**：1.0.0
> **更新日期**：2026-06-28
> **Python版本**：3.11.5
> **硬件支持**：CPU / NVIDIA GPU

Hemispheric_Fingerprints：工具包是一套基于时空深度神经网络与积分梯度特征归因方法的半脑指纹提取工具包。该工具包支持从静息态功能磁共振成像(rs-fMRI)中，进行从交叉验证数据集构建、模型训练，到半脑指纹提取的完整研究流程。具体可参考学位论文《**大脑功能偏侧化的深度神经网络建模与解析**》。

本工具包含两种使用模式：

### 🖥️ GUI交互模式
- **入口文件**：`./Hemispheric_Fingerprints_GUI/dist/Hemispheric_Fingerprints_GUI/Hemispheric_Fingerprints_GUI.exe`
- **硬件需求**：仅需CPU
- **适用场景**：通过图形界面从预训练模型快速生成指纹

### ⚙️ 完整研究流程
- **入口文件**：`./Hemispheric_Fingerprints_Tool`
- **硬件需求**：CPU和GPU
- **适用场景**：需要批量生成大量指纹，端到端流程：数据准备→模型训练→批量指纹提取
---

## 目录

#### 一、环境要求与安装

#### 二、数据预处理与准备

#### 三、第一部分：图形界面工具

#### 四、第二部分：命令行流程

#### 五、故障排除

---

## 一、环境要求与安装

### 1.1 环境要求

| 项目     | 要求                                                                                          |
| -------- | --------------------------------------------------------------------------------------------- |
| 操作系统 | Windows 10                                                                                    |
| 内存     | 16 GB+                                                                                        |
| Python   | 3.11.5                                                                                        |
| GPU      | NVIDIA GPU，显存8GB+（GUI模式下不需要准备）                                                   |
| CUDA     | 与GPU驱动兼容的CUDA版本（GUI模式下不需要准备）                                                |
| 依赖包   | torch, numpy, scikit-learn, captum, nilearn, pandas, openpyxl, PySide6（GUI模式下不需要准备） |

## 二、数据预处理与准备

### 2.1 针对静息态fMRI原始数据的标准预处理流程

本工具包面向经标准最小预处理流程后的静息态fMRI数据。完整预处理流程如下：

1. **梯度非线性校正**：通过梯度非线性对磁场梯度非均匀性引发的空间畸变进行校正；
2. **头动校正**：基于刚体变换模型对头动参数进行估计与补偿，削弱扫描过程中的刚体运动伪影；
3. **几何畸变校正**：利用双向相位编码（LR/RL配对）的拓扑对应关系，对几何畸变进行校正；
4. **CIFTI映射与MNI标准化**：通过CIFTI格式将体素级数据映射至32k灰度坐标空间，并借助非线性配准算法将个体脑影像归一化至MNI152标准模板；
5. **ICA-FIX深度去噪**：采用FSL软件的ICA-FIX框架对最小预处理后的数据进行深度去噪。该算法基于独立成分分析将时序信号解耦为空间成分，并通过机器学习分类器自动识别并剔除与心跳、呼吸周期及微头动相关的结构化噪声成分；

### 2.2 ROI平均时间序列构建

运用左右半球对称性图谱(**AICHA** [Joliot et al., 2015])将大脑皮层分别分割为384个对称区域，并提取每个感兴趣区域(ROI)的平均血氧水平依赖(BOLD)时间序列。

- 对于每个被试，神经影像被构建为 **NC× NT** 的二维矩阵：

  - **NC**：脑区数量（全脑总数，如AICHA为384）
  - **NT**：时间序列长度（扫描时间点数）

### 2.3 半脑样本划分（模型输入格式）

模型以半脑时间序列矩阵（**Nroi × NT**）作为直接输入，其中：

- **Nroi**：单侧脑区数量（AICHA图谱中左右各192）
- **NT**：时间序列长度

具体划分方式：

1. 基于前述构建的全脑 **NC × NT** 矩阵（NC为脑区总数）；
2. 依据图谱的左右半脑标注，将每个被试划分为两个独立样本：
   - **左脑样本**：Nroi × NT
   - **右脑样本**：Nroi × NT
3. **除基本预处理外，不进行功能连接计算、降维或其他特征工程处理**。
4. `session_combined_data.npy` 即为经上述预处理后、按半脑划分好的 **Nroi × NT** 时间序列矩阵堆叠而成的三维数组（被试，时间点，脑区）。

### 2.4 被试和标签数据

- **被试数据**：一维数组 `(N,)`，存储每个样本对应的被试 ID。命名：`session_subject_ids.npy`
- **标签数据**：一维数组 `(N,)`，二分类标签。命名：`session_labels.npy`
  - `0` = 左脑
  - `1` = 右脑

> `N` 为样本总数，两个数组按索引严格对齐。

**注意**：`session_combined_data、session_subject_ids.npy、session_labels.npy`需按照要求命名，且放在同一目录下

## 三、第一部分：可视化工具

本部分面向仅需执行指纹生成的用户，可使用工具自带的预训练模型或自备模型，。提供基于PySide6的图形界面，**仅使用CPU**运行。

### 3.1 启动方式

```bash
运行 ./Hemispheric_Fingerprints_GUI/dist/Hemispheric_Fingerprints_GU/Hemispheric_Fingerprints_GUI.exe
```

### 3.2 界面说明

| 组件                       | 说明                                                                                                                                                       |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **模型目录**         | 存放已训练好的`.pt` 模型文件的文件夹。                                                                                                                   |
| **数据目录**         | 存放代提取数据的文件夹。该目录下应包含三份`.npy` 格式的数据文件                                                                                          |
| **输出目录**         | 指纹文件的保存路径                                                                                                                                         |
| **浏览**             | 点击后打开文件夹选择对话框，选择指定文件夹                                                                                                                 |
| **并行提取样本数量** | 每次从数据集中取多少个样本同时进行特征归因计算。默认值为 4，每增加 1 个单位约额外占用 2GB 内存。建议根据实际硬件配置从较小值开始测试，逐步调整至合适大小。 |
| **生成指纹**         | 点击后开始执行进行特征提取                                                                                                                                 |
| **日志区域**         | 只读文本框，实时显示目录读取状态、文件列表及处理进度等过程                                                                                                 |

### 3.3 操作步骤

1. 点击各路径栏右侧的 **"浏览"** 按钮，分别设置**模型目录**、**数据目录**和**输出目录**；
2. 点击 **"生成指纹"** 按钮，此时按钮自动置灰（如要暂停，则关闭窗口重开）；
3. 等待处理完成，结果自动保存至指定的输出目录；

### 3.4 输出文件

```
hemisphere_feature fingerprint_model_{模型名}_{提取数据名}.npz
```

**输出字段说明**：

| 字段            | 说明                                                    |
| --------------- | ------------------------------------------------------- |
| `features`    | 特征归因二维矩阵（被试，脑区）（时间维度已取中位数降维) |
| `labels`      | 原始标签                                                |
| `subjid`      | 被试ID                                                  |
| `predictions` | 模型预测结果                                            |

> ⚠️ **性能提示**：CPU环境下积分梯度计算速度较慢，建议仅用于小批量验证或无可用的GPU的场景。（生成1份约为50分钟）

---

## 四、第二部分：命令行流程

本部分面向**需要从零开始构建分析流程**的用户，涵盖交叉验证数据集构建、模型训练、性能评估及GPU加速的指纹提取。以下按执行顺序说明。

### 4.1 安装工具包

```bash
# 首次安装
pip install dist/hemispheric_fingerprints_tool-1.01.0-py3-none-any.whl

# 强制重新安装（升级/覆盖已有版本）
pip install --force-reinstall dist/hemispheric_fingerprints_tool-0.0.0-py3-none-any.whl

# 重新打包(修改源码后重新打包可使用该命令)
python setup.py sdist bdist_wheel
```

### 4.2 流程顺序

```
原始fMRI数据
    ↓ 标准预处理（见第二节）
半脑时间序列矩阵 (Nroi × NT)
    ↓ 整理为 session_combined_data.npy / session_subject_ids.npy / session_labels.npy
    ↓ [Step 1] create_cv_datasets
五折交叉验证数据集 (.npz)
    ↓ [Step 2] hcp_cv_model_training
训练好的模型文件 (.pt)
    ↓ [Step 3] evaluate_hcp_session
分类性能报告 (.xlsx)
    ↓ [Step 4] generate_feature_attribution 
特征归因指纹 (.npz)
```

### 4.3 Step 1：构建交叉验证数据集

**功能**：基于被试分组，生成五折交叉验证所需的窗口化数据集。

**调用方式**：

```python
import stdnn_model_train_and_evaluate as st

st.create_cv_datasets(
    data_path,      # 原始 .npy 数据所在目录
    out_path       # 交叉验证数据集输出目录
)
```

**输出文件**：

```
hcp_cross_validation_axis_time_dataset_normz_index_{折数}.npz
```

**输出字段**：

| 字段        | 说明       |
| ----------- | ---------- |
| `x_train` | 训练集特征 |
| `y_train` | 训练集标签 |
| `x_valid` | 验证集特征 |
| `y_valid` | 验证集标签 |
| `x_test`  | 测试集特征 |
| `y_test`  | 测试集标签 |

> **窗口化说明**：工具采用滑动窗口策略（默认窗口大小256，步长64）将长时序数据切分为多个样本，以适配CNN输入格式。

### 4.3 Step 2：模型训练

**功能**：在五折交叉验证数据集上训练时空深度神经网络分类模型。

**调用方式**：

```python
st.hcp_cv_model_training(
    data_path,      # 交叉验证数据集目录（Step 1输出）
    out_path,       # 模型保存目录
    num_workers=4   # 数据加载线程数
)
```

**模型架构**：卷积神经网络（详见 `modelClasses.py`），以半脑时间序列矩阵为输入，进行端到端二分类学习。

**输出模型文件**：

```
model_hcp_aicha_axis_time_normz_window_CV_train_valid_test_window_index_{折数}.pt
```

### 4.4 Step 3：模型性能评估

**功能**：在测试集上评估训练好的模型，生成分类性能报告。

**调用方式**：

```python
st.evaluate_hcp_session(
    model_path, # 模型文件目录
    data_path,  # 验证数据文件目录
    out_path,   # 分析报告保存目录
)
```

**输出文件**：

```
classification_normz_hcp_generalization.xlsx
```

**性能指标**（每折及平均值 ± 标准差）：

| 指标      | 说明          |
| --------- | ------------- |
| Accuracy  | 准确率        |
| Precision | 精确率        |
| Recall    | 召回率        |
| F1-score  | F1分数        |
| AUC       | ROC曲线下面积 |

### 4.5 Step 4：特征归因提取（GPU加速）

**功能**：使用积分梯度法提取每个样本的特征重要性指纹。

**调用方式**：

```python
st.generate_feature_attribution(
    rank,           # 当前进程rank（用于多卡并行）
    world_size,     # 总进程数（GPU数量）
    model_path,    # 模型目录
    data_path,      # 提取数据目录
    out_path       # 输出文件保存目录
)
```

> 🔥 **需要使用GPU加速**。积分梯度计算涉及大量前向/反向传播，多GPU并行可显著缩短计算时间。（3卡GPU下，每份生成时间约100秒）

**输出文件**：

```
hemisphere_feature fingerprint_model_{模型名称}.npz
```

**输出字段**：

| 字段            | 说明                               |
| --------------- | ---------------------------------- |
| `features`    | 特征归因矩阵（结构与输入数据相同） |
| `labels`      | 原始标签                           |
| `subjid`      | 被试ID                             |
| `predictions` | 模型预测结果                       |

### 4.6  超参数配置

模型训练默认参数（可在源码中按需调整）：

| 参数              | 默认值 | 说明             |
| ----------------- | ------ | ---------------- |
| `num_epochs`    | 30     | 最大训练轮数     |
| `batch_size`    | 32     | 批次大小         |
| `learning_rate` | 0.0001 | 初始学习率       |
| `patience_loss` | 8      | 损失早停耐心值   |
| `patience_acc`  | 8      | 准确率早停耐心值 |
| `window_size`   | 256    | 滑动窗口大小     |
| `step`          | 64     | 滑动步长         |

---

## 五、故障排除

如遇到问题，请依次检查以下事项：

1. **Python版本**：确认使用 Python 3.11.5；
2. **依赖包完整性**：确保已安装 `torch`, `numpy`, `scikit-learn`, `captum`, `nilearn`, `pandas`, `openpyxl`
3. **输入数据格式**：确认 `.npy` 文件形状为 `(被试, 时间点, 脑区)`，且与被试ID与标签长度一致并对应；
4. **GPU环境**（如使用）：检查GPU驱动、CUDA版本与PyTorch CUDA版本是否兼容；
5. **文件路径权限**：确保所有输入/输出路径具有读写权限；

---

## 引用与参考文献

- Joliot, M., et al. (2015). AICHA: An atlas of intrinsic connectivity of homotopic areas. *Journal of Neuroscience Methods*, 254, 46–59.
