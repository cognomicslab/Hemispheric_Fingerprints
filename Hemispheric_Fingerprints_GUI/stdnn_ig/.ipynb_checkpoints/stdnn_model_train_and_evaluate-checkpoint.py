from os import path
from sklearn.model_selection import train_test_split, GroupKFold
import numpy as np
import os
import torch
from torch.utils.data import DataLoader,TensorDataset
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support,roc_auc_score
import random
from scipy.interpolate import interp1d
import math
from .modelClasses import ConvNet
from .utilityFunctions import prepare_data_sliding_window, reshapeData, write_excel_file, getInputAttributions
import torch.distributed as dist
import time
from torch.nn.parallel import DistributedDataParallel as DDP
            
def create_cv_datasets(data_list,data_path,out_path,data_name,subject_name,labels_name):
    train_sessions_list  = data_list["train"]
    print(train_sessions_list)
    # 获取长度
    train_len = len(train_sessions_list)
    
    for ii in range(train_len):
        session = train_sessions_list[ii]
        print(session)
        csv_session_folder = data_path + r"/cv_dataset_"+ session
        output_path = out_path + r'/cv_datasets' + r"/cv_dataset_"+ session
        if not os.path.exists(csv_session_folder):
            os.makedirs(csv_session_folder)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_trainlist_index = output_path +r'/train_list_'+ session + '_index' 
        output_testlist_index = output_path +r'/test_list_'+ session + '_index' 
        print(csv_session_folder)
        
        # 读取data数据（多被试的fMRI数据）
        fmri_data_path = csv_session_folder + "/" + data_name
        data = np.load(fmri_data_path)
        print("fmridata.shape:", data.shape)
    
        # 读取被试id
        subjids_path = csv_session_folder + "/" + subject_name
        subjids = np.load(subjids_path)
        print("subjids.shape:",subjids.shape)
    
        labels_path = csv_session_folder + "/" + labels_name
        labels = np.load(labels_path)
        print("labels.shape: ", labels.shape)
        zero_count = np.count_nonzero(labels == 0)
        one_count = np.count_nonzero(labels == 1)
        
        print(f"Left的数量: {zero_count}")
        print(f"Right的数量: {one_count}")
    
        # 初始化StratifiedKFold对象，进行5折分层K折交叉验证。
        # kf = StratifiedKFold(n_splits=5,random_state=None,shuffle=False)
    
        kf = GroupKFold(n_splits=5)
        train_index_list= []
        test_index_list = []
    
        # 遍历K折交叉验证生成的每个分割，将训练和测试索引添加到列表中。
        for train_index,test_index in kf.split(data, groups=subjids):
            train_index_list.append(train_index)
            test_index_list.append(test_index)
    
        np.save(output_trainlist_index, np.array(train_index_list, dtype=object))
        np.save(output_testlist_index, np.array(test_index_list, dtype=object))
    
        np.random.seed(3655) # 设置随机种子，以确保结果的可重复性。
        print('****Preparing windowed data****')
        # 循环5次，为每一折数据创建一个文件名。
        for i in range(5):
            print("the ",i," fold")
            fname = output_path +r'/hcp_cross_validation_axis_time_dataset_'+ session +'_normz_index_%s'%str(i)
            # 根据当前折的训练索引，从数据和标签中选择训练集。
            data_split = data[train_index_list[i]]
            labels_split = labels[train_index_list[i]]
            zero_count = np.count_nonzero(labels_split == 0)
            one_count = np.count_nonzero(labels_split == 1)
            print(f"Left的数量: {zero_count}")
            print(f"Right的数量: {one_count}")
    
            # 将每一折训练集进一步分割为新的训练集和验证集，使用0.1作为验证集的比例。
            # 训练集 (x_train, y_train) 和验证集 (x_valid, y_valid)
            # x特征数据，y标签数据
            x_train,x_valid,y_train,y_valid = train_test_split(data_split,labels_split,
                            test_size=0.2,random_state=67334,stratify=labels_split)
    
            x_train = x_train - np.median(x_train, axis=1, keepdims=True)
            x_valid = x_valid - np.median(x_valid, axis=1, keepdims=True)
            # 为新的训练集和验证集准备窗口化数据。
            window_size = 256
            step = 64
            data_train_window,labels_train_window = prepare_data_sliding_window(x_train,y_train,window_size,step)
            data_valid_window,labels_valid_window = prepare_data_sliding_window(x_valid,y_valid,window_size,step)
            print(data_train_window.shape)
            print(data_valid_window.shape)
    
            # 根据当前折的测试索引，从数据和标签中选择测试集。
            x_test = data[test_index_list[i]]
            y_test = labels[test_index_list[i]]
            x_test = x_test - np.median(x_test, axis=1, keepdims=True)
        
            # 将处理后的数据集保存为一个NumPy存档文件。
            np.savez(fname, x_train = data_train_window, y_train = labels_train_window, x_valid = data_valid_window, y_valid = labels_valid_window, x_test = x_test, y_test = y_test)
        
            print('**Dataset_%s_Saved**'%str(i))
            del data_train_window, data_valid_window, x_train, x_valid, x_test, y_train, y_valid, y_test

def hcp_cv_model_training(data_list,out_path,num_workers):
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    models_sessions_list = data_list["train"]# 定义模型会话列表
    hcp_sessions_list = data_list["test_num"]# 定义HCP会话列表

    # 获取长度
    train_len = len(models_sessions_list)

    cv_session_path = out_path + r'/cv_datasets'
    output_path = out_path + r'/results/classification_excel_files'
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    #输出文件名
    excel_file = output_path + r'/classification_normz_within_session.xlsx'
    best_split_file = output_path + r'/best_performance_split.npz'
    
    if os.path.exists(excel_file):
        os.remove(excel_file)
        print(f"已删除旧的Excel文件: {excel_file}")
        
    # Hyperparameters
    num_epochs = 30
    num_classes = 2
    batch_size = 32
    learning_rate = 0.0001
    patience_loss = 8
    patience_acc  = 8

    USE_PRETRAINED_MODEL = False # 是否使用预训练模型
    use_cuda = True #是否使用cuda
    
    # 保存每个session的最佳批次
    best_split_id = np.empty((0,1),int)
    
    
    for ii in range(train_len): 
        session = models_sessions_list[ii] 
        print(session)
        path_to_dataset = cv_session_path + r"/cv_dataset_" + session 
        path_to_output = out_path + r'/outputs/saved_models/models_cv_' + session 
      
        if not os.path.exists(path_to_output):
            os.makedirs(path_to_output)
    
        model_ss = hcp_sessions_list[ii]  
        test_ss = hcp_sessions_list[ii] 
        
        # 保存 acc precision recall f1
        test_acc = np.empty((0,1),float)
        precision = np.empty((0,1),float)
        recall = np.empty((0,1),float)
        f1_score = np.empty((0,1),float)
        auc_values = np.empty((0,1),float)
    
        for m in range(5):
            # 五折数据
            fname_dataset = path_to_dataset + r'/hcp_cross_validation_axis_time_dataset_' + session + r'_normz_index_%s.npz'%str(m)
   
            fname_model = path_to_output + r'/model_hcp_aicha_axis_time_' + session + r'_normz_window_CV_train_valid_test_window_index_%s.pt' % str(m)
           
            print("file names: \n dataset {}\n output model {}\n".format(fname_dataset, fname_model))
            datao  = np.load(fname_dataset)
    
            print("Data Processing")
            # 处理训练数据
            print("train data")
            x_train = datao['x_train']
            y_train = datao['y_train'].astype('int64')
            x_train = reshapeData(x_train)
            
            # 处理验证数据
            print("valid data")
            x_valid = datao['x_valid']
            y_valid = datao['y_valid'].astype('int64')
            x_valid = reshapeData(x_valid)
            
            # 处理测试数据
            print("test data")
            x_test = datao['x_test']
            y_test = datao['y_test'].astype('int64')
            x_test = reshapeData(x_test)
    
            print("PyTorch into TensorDataset")
            # 测试数据
            input_sensor = torch.from_numpy(x_train).type(torch.FloatTensor)
            label_tensor = torch.from_numpy(y_train)
            dataset_train1 = TensorDataset(input_sensor,label_tensor)
    
            # 验证数据
            input_tensor_valid = torch.from_numpy(x_valid).type(torch.FloatTensor)
            label_tensor_valid = torch.from_numpy(y_valid)
            dataset_valid1 = TensorDataset(input_tensor_valid, label_tensor_valid)
            
            # 测试数据
            input_tensor_test = torch.from_numpy(x_test).type(torch.FloatTensor)
            label_tensor_test = torch.from_numpy(y_test)
            dataset_test1 = TensorDataset( input_tensor_test, label_tensor_test )
            
            # 加载测试和训练数据
            print("Load Train and Test data into the loader")
            train_loader = DataLoader(dataset=dataset_train1,batch_size=batch_size,shuffle=True, num_workers=num_workers,pin_memory=True,persistent_workers=True)
            valid_loader = DataLoader(dataset=dataset_valid1, batch_size=batch_size, shuffle=False, num_workers=num_workers,pin_memory=True,persistent_workers=True)
            test_loader = DataLoader(dataset=dataset_test1, batch_size=x_test.shape[0], shuffle=False, num_workers=num_workers,pin_memory=True,persistent_workers=True)
    
            # 定能一模型
            model = ConvNet()
    
            # 如果使用的cuda将模型移至GPU
            if use_cuda and torch.cuda.is_available():
                model.cuda()
    
            
            if USE_PRETRAINED_MODEL:
                print("Using the existing trained model")
                model.load_state_dict(torch.load(fname_model))
            else:
                print("Training the model")
                # Loss and optimizer
                print("Loss and optimizer")
                # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
                # criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
                criterion = nn.CrossEntropyLoss()
                # criterion = FocalLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-4)
                # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=5e-3)
                # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                best_val_loss = float('inf')
                best_val_acc = 0.0
    
                print("patience_loss:{};patience_acc:{}".format(patience_loss,patience_acc))
                
                # 初始化早停计数器
                epochs_no_improve_loss = 0
                epochs_no_improve_acc = 0
    
         
                total_step = len(train_loader)
                acc_list =[]
                val_acc_temp = 0.0
    
                #  开始训练模型
                print("Start Training")
                for epoch in range(num_epochs):
                    print("start each epoch")
                    print("start model train")
                    model.train()
    
                    correct = 0
                    total = 0
                    for i , (data_ts,labels) in enumerate(train_loader):
                        # print("start train_loader: "+ str(i))
                        data_ts = data_ts.cuda()
                        labels = labels.cuda()
    
                        # data_ts = data_ts + torch.randn_like(data_ts) * 0.05  # 5%噪声
                        
                        # Run the forward pass
                        outputs = model(data_ts)
                        loss = criterion(outputs,labels)
                        
                        # 反向传播与性能优化
                        optimizer.zero_grad() #清空所有参数的梯度。每个批次开始前，都要从干净的状态开始计算梯度。
                        loss.backward() # 从最终的损失值反向传播，计算每个参数对损失的贡献度（梯度）
                        optimizer.step()
                        # total_loss += loss.item()
                        
                        # 追溯模型性能
                        _,predicted = torch.max(outputs.data,1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                                         
                        # 打印训练进程
                        if (i + 1) % 10 == 0:
                            print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Accuracy: {:.2f}%'
                                .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                                        (correct / total) * 100))
    
                    train_acc = correct / total
                    # avg_train_loss = total_loss / len(train_loader)
                    # acc_list.append(100 * correct / total_samples)
                    
                    print("start model eval")
                    # 设置模型为评估模式，并初始化准确率计算变量。
                    model.eval()
                    with torch.no_grad():
                        correct = 0
                        total = 0
    
                        val_loss = 0.0  
                        
                        # eval model in valid data
                        print("eval model in valid data")
                        for images, labels in valid_loader:
                            # print(images.shape, labels.shape)
                            if use_cuda and torch.cuda.is_available():
                                images = images.cuda()
                                labels = labels.cuda()
                            outputs = model(images) 
                            _, predicted = torch.max(outputs.data,1)
                            loss = criterion(outputs, labels) 
                            val_loss += loss.item()
                            
                            # Accumulate  total validation data and num of correct predictions.
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
    
                        avg_val_loss = val_loss / len(valid_loader)  # 平均验证损失
                        val_acc = correct / total
                        print('Validation Loss of the model on the Val data: {} '.format(avg_val_loss))
                        print('Validation Accuracy of the model on the Val data: {} %'.format((correct / total) * 100))

        
                    if avg_val_loss < best_val_loss and val_acc > best_val_acc:
                        best_val_loss = avg_val_loss
                        best_val_acc = val_acc
                        epochs_no_improve_loss = 0
                        epochs_no_improve_acc = 0
                        # 保存最佳模型
                        print('**Saving Model on Drive**')
                        torch.save(model.state_dict(), fname_model)
                    else:
                        if avg_val_loss >= best_val_loss:
                            epochs_no_improve_loss += 1
                        if val_acc <= best_val_acc:
                            epochs_no_improve_acc += 1
    
                    # 检查是否触发早停
                    # if epochs_no_improve_loss >= patience_loss or epochs_no_improve_acc >= patience_acc or val_acc >= 0.995 and epoch > 3:
                    # if  val_acc >= 0.995 and train_acc >= 0.995 and epoch >= 3:
                    if  val_acc >= 0.95 and train_acc >= 0.95 and epoch > 5:
                        print(f'Early stopping triggered after {epoch + 1} epochs!')
                        break
                
        
            # Load the saved model weights
            model.load_state_dict(torch.load(fname_model))
            print("start fold m model eval")
            model.eval()
            with torch.no_grad():
                probabilities_list = []
                labels_list = []
                
                correct = 0
                total = 0
                for images, labels in test_loader:
                    if use_cuda and torch.cuda.is_available():
                        images = images.cuda()
                        labels = labels.cuda()
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    probabilities = torch.softmax(outputs, dim=1)[:, 1]
                    probabilities_list.extend(probabilities.cpu().numpy())
                    labels_list.extend(labels.cpu().numpy())
    
                print('Test Accuracy of the model on the  test data: {} %'.format((correct / total) * 100))
                test_acc = np.append(test_acc, 100 * correct / total)
            
            # print fold m results
            print(classification_report(labels.detach().cpu(),predicted.detach().cpu()))
    
            report = precision_recall_fscore_support(labels.detach().cpu(),predicted.detach().cpu())
            print("precision: {},{}\n".format(np.mean(report[0].round(2)),np.std(report[0].round(2))))
            print("recall: {},{}\n".format(np.mean(report[1].round(2)), np.std(report[1].round(2))))
            print("f1-score: {},{}\n".format(np.mean(report[2].round(2)), np.std(report[2].round(2))))
    
            # current fold
            precision = np.append(precision, np.mean(report[0]))
            recall = np.append(recall, np.mean(report[1]))
            f1_score = np.append(f1_score, np.mean(report[2]))
            
            # 计算当前折的AUC值
            current_auc = roc_auc_score(labels_list, probabilities_list)
            print('Test AUC of the model on the test data: {:.4f}'.format(current_auc))
            # 将当前折的AUC值添加到数组中
            auc_values = np.append(auc_values, current_auc)
            
            print("Confusion Matrix:")
            print (confusion_matrix(labels.detach().cpu(),predicted.detach().cpu()))
        
       
        # print all fold
        print("test accuracy (mean, std): {}, {}\n".format(np.mean(test_acc), np.std(test_acc)))
        print("precision (mean, std): {}, {}\n".format(np.mean(precision), np.std(precision)))
        print("recall (mean, std): {}, {}\n".format(np.mean(recall), np.std(recall)))
        print("f1_score (mean, std): {}, {}\n".format(np.mean(f1_score), np.std(f1_score)))
        print("auc (mean, std): {}, {}\n".format(np.mean(auc_values), np.std(auc_values)))

        if USE_PRETRAINED_MODEL: # write to excel only if model has been well trained
            write_excel_file(test_acc, precision, recall, f1_score, excel_file, model_ss, test_ss,auc_values)
            print(excel_file)
            
        write_excel_file(test_acc, precision, recall, f1_score, excel_file, model_ss, test_ss,auc_values)
        print("{} 文件保存".format(session))
        # break
    
    print(hcp_sessions_list)
    print(best_split_id)
    np.savez(best_split_file, sessions=hcp_sessions_list, best_split_id = best_split_id)

def evaluate_hcp_session(data_list,out_path,data_path,data_name,subject_name,labels_name):
    def format_no_round(value):
        value = int(value*1000)/1000
        return f"{value:.3f}"

    models_sessions_list = data_list["train"]
    test_sessions_list = models_sessions_list[:] 

    # 获取长度
    train_len = len(models_sessions_list)
    test_len = len(test_sessions_list)

    output_path = out_path + r'/results/classification_excel_files'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    excel_file = output_path + r'/classification_normz_hcp_generalization.xlsx'
    if os.path.exists(excel_file):
        os.remove(excel_file)
    print(f"已删除旧的Excel文件: {excel_file}")
    
    # 在其他三个扫描数据上评估模型性能
    for ii in range(train_len): 
        # print(ii)
        models_session = models_sessions_list[ii] 
        print(models_session)
    
        for jj in range(test_len):
            # print(jj)
            test_session = test_sessions_list[jj]
            if test_session == models_session:
                continue
            path_to_models = out_path + r'/outputs/saved_models/models_cv_' + models_session 
            path_to_data = data_path +  r'/cv_dataset_'+ test_session
 
            print(path_to_models)
            print(test_session)
            print(path_to_data)
            testdata = path_to_data +  '/' + data_name
            print("apply models (from {}) to test data {}\n".format(models_session, test_session))
            
            
            test_acc = np.empty((0,1),float)
            precision = np.empty((0,1),float)
            recall = np.empty((0,1),float)
            f1_score = np.empty((0,1),float)
            auc_values = np.empty((0,1),float)
            
            # Load test data and data cleaning etc.
            print("data loading")
            data = np.load(testdata)
            data = data - np.median(data, axis=1, keepdims=True)
            print("data.shape: ", data.shape)
                    
            # subjids and labels
            # 读取被试id
            subjids_path = path_to_data +  '/' + subject_name
            subjids = np.load(subjids_path)
            print("subjids.shape: ",subjids.shape)
            # print(subjids)
    
            # 读取 被试信息 文件
            labels_path = path_to_data +  '/' + labels_name
            labels = np.load(labels_path)
            print("labels.shape: ", labels.shape)
            # print(labels)
            
            print("total number of subjects: {}\n".format(len(labels)))
            print("female/male: {}/{}\n".format(sum(labels), len(labels)-sum(labels)))
            
            x_test = data
            print('before interpolation')
            print(x_test.shape)
            print('after interpolation')
            x_test = reshapeData(x_test)
            print(x_test.shape)
            
            y_test = labels      
            print('Data loading completed')
            
            input_tensor_test = torch.from_numpy(x_test).type(torch.FloatTensor)
            label_tensor_test = torch.from_numpy(y_test)
            dataset_test1 = TensorDataset(input_tensor_test, label_tensor_test)
            
            # Load Test data into the loader
            test_loader = DataLoader(dataset=dataset_test1, batch_size=x_test.shape[0], shuffle=False, num_workers=4)
            
            num_classes = 2
            for m in range(5):
                fname_model = path_to_models + r'/model_hcp_aicha_axis_time_' + models_session + '_normz_window_CV_train_valid_test_window_index_%s.pt' % str(m)
            
                print("model name: {}".format(fname_model))
                
                # define model
                model = ConvNet()
                USE_PRETRAINED_MODEL = True
                use_cuda = True 
                
                if USE_PRETRAINED_MODEL:
                    print("Using the existing trained model")
                    model.load_state_dict(torch.load(fname_model))
    
                if use_cuda and torch.cuda.is_available():
                    model.cuda()
                
                model.eval()
                with torch.no_grad():
                    probabilities_list = []
                    labels_list = []
                    pred_list = []
                    
                    correct = 0
                    total = 0
                    for images, labels in test_loader:
                        if use_cuda:
                            images = images.cuda()
                            labels = labels.cuda()
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        
                        probabilities = torch.softmax(outputs, dim=1)[:, 1]
                        probabilities_list.extend(probabilities.cpu().numpy())
                        labels_list.extend(labels.cpu().numpy())
                        pred_list.extend(predicted.cpu().numpy())
    
                    print('Test Accuracy of the model on the  test data: {} %'.format((correct / total) * 100))
                    test_acc = np.append(test_acc, 100 * correct / total)
                
                # print fold m results
                print(classification_report(labels.detach().cpu(),predicted.detach().cpu()))
                
                # print avg results for a fold
                # report = precision_recall_fscore_support(labels.detach().cpu(), predicted.detach().cpu())
                report = precision_recall_fscore_support(labels_list, pred_list)
                print(report)
                print("precision: {},{}\n".format(np.mean(report[0]), np.std(report[0])))
                print("recall: {},{}\n".format(np.mean(report[1]), np.std(report[1])))
                print("f1-score: {},{}\n".format(np.mean(report[2]), np.std(report[2])))
                
                precision = np.append(precision, np.mean(report[0]))
                recall = np.append(recall, np.mean(report[1]))
                f1_score = np.append(f1_score, np.mean(report[2]))
    
                # 计算当前折的AUC值
                current_auc = roc_auc_score(labels_list, probabilities_list)
                print('Test AUC of the model on the test data: {:.4f}'.format(current_auc))
                # 将当前折的AUC值添加到数组中
                auc_values = np.append(auc_values, current_auc)
    
                print("Confusion Matrix:")
                print(confusion_matrix(labels.detach().cpu(),predicted.detach().cpu()))
    
            print(auc_values)
            
            # print results averaged across folds
            print("test accuracy (mean, std): {}, {}\n".format(np.mean(test_acc), np.std(test_acc)))
            print("precision (mean, std): {}, {}\n".format(np.mean(precision), np.std(precision)))
            print("recall (mean, std): {}, {}\n".format(np.mean(recall), np.std(recall)))
            print("f1_score (mean, std): {}, {}\n".format(np.mean(f1_score), np.std(f1_score)))
            print("auc (mean, std): {}, {}\n".format(np.mean(auc_values), np.std(auc_values)))
    
            write_excel_file(test_acc, precision, recall, f1_score, excel_file, models_session, test_session,auc_values)
            
            print(excel_file)
        #     break
        # break
    
def generate_feature_attribution(rank, world_size,data_path,out_path,data_list):
    # 设置当前进程使用的 GPU
    torch.cuda.set_device(rank)
    # 初始化分布式环境
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    project_path = out_path

    output_path = project_path +  r'/results/attributions/indiv_cv'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model_sessions_list = data_list["train"]
    test_sessions_list = data_list["test"]
    train_len = len(model_sessions_list)
    test_len = len(test_sessions_list)
    
    # for ii in range(8): 
    for ii in range(train_len): 
        for tt in range(test_len):
            model_session = model_sessions_list[ii]
            test_session = test_sessions_list[tt]
            
            path_to_dataset = data_path + r"/cv_dataset_" + test_session 
            print('model session: {}; test session: {}'.format(model_session, test_session))
            print(path_to_dataset)
            # load &  data
            fmri_path = path_to_dataset +  r'/session_combined_data.npy'
            data = np.load(fmri_path)
            data = data - np.median(data, axis=1, keepdims=True)
    
            # subjids and labels
            subjids_path = path_to_dataset + r'/session_subject_ids.npy'
            subjid = np.load(subjids_path)
            print("subjids.shape: ",subjid.shape)
            # print(subjids)
    
            gender_path = path_to_dataset + r'/session_labels.npy'
            labels = np.load(gender_path)
            print("labels.shape: ", labels.shape)
    
            print('Data loading completed')
            
            print("test data dimension before reshape: {}".format(data.shape))
            data = reshapeData(data)
            print("test data dimension after reshape: {}".format(data.shape))
            
            # prepare test data for data loader
            input_tensor_test = torch.from_numpy(data).type(torch.FloatTensor)
            label_tensor_test = torch.from_numpy(labels)
            subjid_tensor_test = torch.from_numpy(subjid)
            dataset_test = TensorDataset(input_tensor_test, label_tensor_test)
    
            # load test data into the loader
            test_loader = DataLoader(dataset=dataset_test, batch_size=data.shape[0], shuffle=False, num_workers=12,pin_memory=True,persistent_workers=True)
    
            for m in range(5): # 5 splits/folds (5 models); get features attributions for each model
                path_to_model= project_path + r'/outputs/saved_models/models_cv_' + model_session  
                fname_model = path_to_model + r'/model_hcp_aicha_axis_time_' + model_session +'_normz_window_CV_train_valid_test_window_index_%s.pt' % str(m)
                print("model name: {}".format(fname_model))
            
                model = ConvNet()
                model.load_state_dict(torch.load(fname_model))
            
                use_cuda = True #False
                if use_cuda and torch.cuda.is_available():
                    if torch.cuda.device_count() > 1:
                        print("Let's use", torch.cuda.device_count(), "GPUs!")
                        model = model.to(rank)
                        # Wrap the model with DataParallel
                        # model = nn.DataParallel(model)
                        model = DDP(model, device_ids=[rank], output_device=rank)
                    else:
                        model = model.to(rank)
                   
                
    
                # making prediction on test data
                model.eval()
                predictions = []
                with torch.no_grad():
                    correct = 0
                    total = 0
                    # for images, labels in test_loader:
                    for i, (images, labels) in enumerate(test_loader):
                        print(f"Processing batch {i + 1}/{len(test_loader)} on rank {rank}")
            
                        # 检查输入数据的形状和类型
                        print(f"Images shape: {images.shape}, dtype: {images.dtype}")
                        print(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")
                        
                        # 将数据移动到 GPU
                        if use_cuda:
                            
                            # 将数据移动到当前 GPU
                            images = images.to(rank)
                            labels = labels.to(rank)
                            # images = images.cuda()
                            # labels = labels.cuda()
                        
                        # 模型推理
                        print("Running model inference")
                        outputs = model(images)
                        print("Model inference completed")
                        
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        predictions.append(predicted.cpu().detach().numpy())
                    predictions = np.concatenate(predictions)
                    print('Split/Fold {}: Test Accuracy of the model on the  test data: {} %'.format(str(m), (correct / total) * 100))
    
            
                '''Feature Attributions (on test data)'''
                input_tensor_test = input_tensor_test.cuda() # do not use cuda as it leads to runtime error
                print(label_tensor_test.size())
                
                features = []    
                # for i in range(len(input_tensor_test)):
                #     # print(label_tensor_test[i], type(label_tensor_test[i]))
                #     attr = getInputAttributions(model, input_tensor_test[i].unsqueeze_(-1).permute(2,0,1), label_tensor_test[i]) # female = 1; male = 0
                #     features.append(attr)

                batch_size = 32
                for i in range(0, len(input_tensor_test), batch_size):
                    batch = input_tensor_test[i:i+batch_size]
                    batch_labels = label_tensor_test[i:i+batch_size]
                    
                    for j in range(len(batch)):
                        attr = getInputAttributions(model, batch[j].unsqueeze_(-1).permute(2,0,1), batch_labels[j])
                        features.append(attr)
                    if i % 100 == 0:
                        print("i:",i)
                features = np.concatenate(features)
    
                feature_attribution_fname = output_path + '/hcp_model_' + model_session + '_index_' + str(m) + '_test_' + test_session + '.npz'
    
                np.savez(feature_attribution_fname, features=features, labels=label_tensor_test.numpy(), subjid=subjid_tensor_test.numpy(), predictions=predictions)
            #     break
            # break
        # 清理分布式环境
        dist.destroy_process_group()

def generate_feature_attribution_cpu(data_path,out_path,data_list):
    # 初始化分布式环境
    project_path =  out_path
    output_path = project_path +  r'/results/attributions/indiv_cv'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model_sessions_list = data_list["train"]
    test_sessions_list = data_list["test"]
    train_len = len(model_sessions_list)
    test_len = len(test_sessions_list)
    
    # for ii in range(8): 
    for ii in range(train_len): 
        for tt in range(test_len):
            model_session = model_sessions_list[ii]
            test_session = test_sessions_list[tt]
            
            path_to_dataset = data_path + r"/cv_dataset_" + test_session 
            print('model session: {}; test session: {}'.format(model_session, test_session))
            # load &  data
            fmri_path = path_to_dataset +  r'/session_combined_data.npy'
            data = np.load(fmri_path)
            data = data - np.median(data, axis=1, keepdims=True)
    
            # subjids and labels
            subjids_path = path_to_dataset + r'/session_subject_ids.npy'
            subjid = np.load(subjids_path)
            print("subjids.shape: ",subjid.shape)
            # print(subjids)
    
            gender_path = path_to_dataset + r'/session_labels.npy'
            labels = np.load(gender_path)
            print("labels.shape: ", labels.shape)
    
            print('Data loading completed')
            
            print("test data dimension before reshape: {}".format(data.shape))
            data = reshapeData(data)
            print("test data dimension after reshape: {}".format(data.shape))
            
            # prepare test data for data loader
            input_tensor_test = torch.from_numpy(data).type(torch.FloatTensor)
            label_tensor_test = torch.from_numpy(labels)
            subjid_tensor_test = torch.from_numpy(subjid)
            dataset_test = TensorDataset(input_tensor_test, label_tensor_test)
    
            # load test data into the loader
            test_loader = DataLoader(dataset=dataset_test, batch_size=data.shape[0], shuffle=False, num_workers=6,pin_memory=False,persistent_workers=True)
    
            for m in range(5): # 5 splits/folds (5 models); get features attributions for each model
                path_to_model= project_path + r'/outputs/saved_models/models_cv_' + model_session  
                fname_model = path_to_model + r'/model_hcp_aicha_axis_time_' + model_session +'_normz_window_CV_train_valid_test_window_index_%s_recorrect.pt' % str(m)
                print("model name: {}".format(fname_model))
            
                model = ConvNet()
                model.load_state_dict(torch.load(fname_model, map_location='cpu'))
            
                # making prediction on test data
                model.eval()
                predictions = []
                with torch.no_grad():
                    correct = 0
                    total = 0
                    # for images, labels in test_loader:
                    for i, (images, labels) in enumerate(test_loader):
                        print(f"Processing batch {i + 1}/{len(test_loader)} ")

                        # 模型推理
                        print("Running model inference")
                        outputs = model(images)
                        print("Model inference completed")
                        
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        predictions.append(predicted.cpu().detach().numpy())
                    predictions = np.concatenate(predictions)
                    print('Split/Fold {}: Test Accuracy of the model on the  test data: {} %'.format(str(m), (correct / total) * 100))
    
            
                '''Feature Attributions (on test data)'''
                input_tensor_test = input_tensor_test# do not use cuda as it leads to runtime error
                print(label_tensor_test.size())

                features = []  
                input_n = len(input_tensor_test)
                # for i in range(input_n):
                #     # print(label_tensor_test[i], type(label_tensor_test[i]))
                #     attr = getInputAttributions(model, input_tensor_test[i].unsqueeze_(-1).permute(2,0,1), label_tensor_test[i]) 
                #     features.append(attr)
                #     if i % 100 == 0:
                #         print("{}/{}".format(i,input_n))

                start = time.time()
                batch_size = 5  # 根据显存调整，GPU 可设 64-128
                for i in range(0, input_n, batch_size):
                    batch = input_tensor_test[i:i+batch_size]  # [B, 192, 1200]
                    targets = label_tensor_test[i:i+batch_size]  # [B]
                    
                    attr = getInputAttributions(model, batch, targets)  # [B, 192, 1200]
                    features.append(attr)
                    
                    if i % 100 == 0:
                        print("{}/{}".format(i, input_n))

                print(f"耗时: {time.time()-start:.2f}s")
                features = np.concatenate(features)
                print("len(features):",len(features))
                feature_attribution_fname = output_path + '/hcp_model_' + model_session + '_index_' + str(m) + '_test_' + test_session + '.npz'
    
                np.savez(feature_attribution_fname, features=features, labels=label_tensor_test.numpy(), subjid=subjid_tensor_test.numpy(), predictions=predictions)
                break
            break
        break
        
