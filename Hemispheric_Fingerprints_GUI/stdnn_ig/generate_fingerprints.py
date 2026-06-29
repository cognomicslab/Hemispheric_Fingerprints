import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader,TensorDataset
from scipy.interpolate import interp1d
from .modelClasses import ConvNet
from .utilityFunctions import reshapeData, getInputAttributions
import torch.distributed as dist
import time
import psutil
import gc
import re

def generate_feature_attribution_cpu(model_path,data_path, output_path, batch,stop_check=None):
    print("开始生成指纹！")
    batch_size = batch
    # 初始化分布式环境
    # 检查可用内存
    available_memory = psutil.virtual_memory().available
    print(f"可用内存: {available_memory / (1024**3):.2f} GB")
    print("模型目录：{}".format(model_path))
    print("数据目录：{}".format(data_path))
    print("输出目录：{}".format(output_path))

    # 读取 所有模型文件
    models_files = [os.path.splitext(f)[0] for f in os.listdir(model_path) if f.endswith('.pt')]
    print("文件夹下所有文件名：",models_files)
    model_num = len(models_files)
    # for ii in range(8): 
    for ii in range(model_num): 
        fname_model = model_path + r'/' + models_files[ii] + '.pt'
        print("模型名称: {}".format(fname_model))
        data_name = re.search(r'/([^/]+)$', data_path).group(1)

        for tt in range(1):
            # load &  data
            fmri_path = data_path +  r'/session_combined_data.npy'
            data = np.load(fmri_path)
            data = data - np.median(data, axis=1, keepdims=True)
    
            # subjids and labels
            subjids_path = data_path + r'/session_subject_ids.npy'
            subjid = np.load(subjids_path)
            print("被试数据维度: ",subjid.shape)
            # print(subjids)
    
            labels_path = data_path + r'/session_labels.npy'
            labels = np.load(labels_path)
            print("标签数据维度.shape: ", labels.shape)
    
            # print('Data loading completed')
            
            print("fMRI数据维度: {}".format(data.shape))
            # print("test data dimension before reshape: {}".format(data.shape))
            data = reshapeData(data)
            # print("test data dimension after reshape: {}".format(data.shape))
            
            # prepare test data for data loader
            input_tensor_test = torch.from_numpy(data).type(torch.FloatTensor)
            label_tensor_test = torch.from_numpy(labels)
            if np.issubdtype(subjid.dtype, np.number):
                subjid_tensor_test = torch.from_numpy(subjid)
            else:
                subjid_tensor_test = subjid
            dataset_test = TensorDataset(input_tensor_test, label_tensor_test)
    
            # load test data into the loader
            # test_loader = DataLoader(dataset=dataset_test, batch_size=data.shape[0], shuffle=False, num_workers=6,pin_memory=False,persistent_workers=True)
            test_loader = DataLoader(dataset=dataset_test, batch_size=16, shuffle=False, num_workers=0,pin_memory=False,persistent_workers=False)
            
            
            model = ConvNet()
            model.load_state_dict(torch.load(fname_model, map_location='cpu'))
            
            # making prediction on test data
            model.eval()
            predictions = []
            with torch.no_grad():
                correct = 0
                total = 0
                # for images, batch_labels in test_loader:
                for i, (images, batch_labels) in enumerate(test_loader):
                    if i == 0:
                        print("正在运行模型推理")
                    outputs = model(images)
                               
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
                    predictions.append(predicted.cpu().detach().numpy())
                predictions = np.concatenate(predictions)
                print('准确率：{:.2f}%'.format((correct / total) * 100))
    
            '''Feature Attributions (on test data)'''
            input_tensor_test = input_tensor_test# do not use cuda as it leads to runtime error
            # print(label_tensor_test.size())

            features = []  
            input_n = len(input_tensor_test)

            start = time.time()
            print("batch:{},并行提取{}个样本".format(batch_size,batch_size))
            # for i in range(0, input_n, batch_size):
            for i in range(0, input_n, batch_size):
                batch = input_tensor_test[i:i+batch_size]  # [B, 192, 1200]
                targets = label_tensor_test[i:i+batch_size]  # [B]

                attr = getInputAttributions(model, batch, targets)  # [B, 192, 1200]
                attr_median = np.median(attr, axis=2)
                # print(attr_median.shape)
                features.append(attr_median)  
                
                if  (i + batch_size) % 100 == 0 :
                    print("batch:{}/{}".format(i + batch_size, input_n))


            print(f"耗时: {time.time()-start:.2f}s")
            features = np.concatenate(features)
            excel_path = output_path + '/hemisphere_feature_fingerprint_model_{}_{}.xlsx'.format(models_files[ii], data_name)
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                pd.DataFrame(features).to_excel(writer, sheet_name='features', index=False, header=False)
                pd.DataFrame(labels).to_excel(writer, sheet_name='labels', index=False, header=False)
                pd.DataFrame(subjid).to_excel(writer, sheet_name='subjid', index=False, header=False)
                pd.DataFrame(predictions).to_excel(writer, sheet_name='predictions', index=False, header=False)

            # feature_attribution_fname = output_path + '/hemisphere_feature fingerprint_model_{}_{}.npz'.format(models_files[ii],data_name)
            # np.savez(feature_attribution_fname, features=features, labels=label_tensor_test.numpy(), subjid=subjid_tensor_test.numpy(), predictions=predictions)
            
            print("✅ 生成指纹文件：{}".format(excel_path))
            del model, input_tensor_test, label_tensor_test, subjid_tensor_test
            del dataset_test, test_loader, predictions, features
            gc.collect()
            break
        break
    print("\n✅ 指纹生成完成")
        