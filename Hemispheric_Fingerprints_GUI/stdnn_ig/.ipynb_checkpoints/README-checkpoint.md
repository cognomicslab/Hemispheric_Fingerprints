推荐配置环境
Python 3.11.5

1.安装工具包
pip install dist/stdnn_ig-0.1.0-py3-none-any.whl 或
pip install --force-reinstall dist/stdnn_ig-0.1.0-py3-none-any.whl 

2.调用函数
可参考 demonstration.py
create_cv_datasets:生成五折交叉验证所需数据
    输入：fMRI文件、被试id文件、分类标签文件
    输出：五折交叉数据集（hcp_cross_validation_axis_time_dataset_LR_S1_normz_index_0.npz）
hcp_cv_model_training:训练分类模型
    输入：五折交叉验证数据集
    输出：模型（model_hcp_aicha_axis_time_LR_S1_normz_window_CV_train_valid_test_window_index_0.pt）
evaluate_hcp_session:模型性能评估
    输入：训练好的模型
    输出：模型性能表格
generate_feature_attribution:使用IG方法提取特征归因提取样本指纹
    输出：模型、所提取的数据集
    输出：与输入矩阵结构相同的特征归因文件（hcp_model_LR_S1_index_0_test_LR_S1.npz）
    

