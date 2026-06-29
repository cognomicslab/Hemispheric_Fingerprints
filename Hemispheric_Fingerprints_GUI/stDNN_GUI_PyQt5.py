import time
print("0. 开始导入模块...", time.time())
import sys
from PySide6.QtWidgets  import QApplication, QMainWindow, QLabel, QWidget, QHBoxLayout, QPushButton, QLineEdit,QPlainTextEdit, QFileDialog, QVBoxLayout,QTextEdit
from PySide6.QtCore import QThread, Signal, QObject
from stdnn_ig import generate_fingerprints as st
import os
print("1. PySide6导入完成:", time.time())

def get_app_dir():
    """获取exe所在目录（打包后也能正确指向exe同级目录）"""
    if getattr(sys, 'frozen', False):
        # 打包后的exe运行环境
        return os.path.dirname(sys.executable)
    else:
        # 直接运行.py
        return os.path.dirname(os.path.abspath(__file__))
    
# 全局停止标志
_stop_flag = False

# 在 MyWindow 类定义之前添加这个类
class TextRedirector(QObject):
    text_ready = Signal(str)
    
    def __init__(self, text_widget):
        super().__init__()
        self.text_ready.connect(text_widget.append)
    
    def write(self, text):
        if text.strip():
            self.text_ready.emit(text.strip())
    
    def flush(self):
        pass

# 在工作线程类中添加停止检查
class WorkerThread(QThread):
    error = Signal(str)
    stopped = Signal() 
    
    def __init__(self, model_path,data_path, output_path, batch):
        super().__init__()
        self.model_path = model_path
        self.data_path = data_path
        self.output_path = output_path
        self.batch = batch 
        self._stop_flag = False
        self._is_stopped = False 
    
    def stop(self):
        self._stop_flag = True
    
    def run(self):
        try:
            # 重定向输出
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = TextRedirector(window.log_text)
            sys.stderr = TextRedirector(window.log_text)
            
            try:
                # 调用函数，传入停止检查
                # print("DEBUG: 开始调用函数")
                st.generate_feature_attribution_cpu(
                    model_path=self.model_path,
                    data_path=self.data_path,
                    output_path=self.output_path,
                    batch=self.batch,
                    stop_check=lambda: self._stop_flag
                )
                # print("DEBUG: 函数正常返回")
                # 正常完成
            except InterruptedError as e:  # ✅ 专门捕获停止异常
                # 检查是否是停止导致的异常
                if self._stop_flag:
                    print("DEBUG: 检测到停止标志，发射stopped信号")
                    self.stopped.emit()
                else:
                    print(f"DEBUG: 其他异常: {e}")
                    self.error.emit(str(e))
            finally:
                # print("DEBUG: finally 块执行")
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                
        except Exception as e:
            print(f"DEBUG: 其他异常: {e}")
            self.error.emit(str(e))
        # print("DEBUG: run 结束")

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()  # 调用界面初始化函数
        self.initEvents()  # 调用事件/信号初始化函数（可选）
        self.worker = None  # 工作线程
        global _stop_flag  
        _stop_flag = False  


    def initUI(self):
        """初始化界面"""
        self.setWindowTitle('特征生成器')
        self.setGeometry(500, 300, 900, 600) 
        
        # 创建中央部件（先创建这个变量）
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 垂直分布
        layout = QVBoxLayout(central_widget)

        # 第一行 模型数据获取
        row1_widget = QWidget()  # 创建容器
        row1_layout = QHBoxLayout(row1_widget) #水平布局
        row1_layout.addWidget(QLabel('模型目录:'))
        self.input_box1 = QLineEdit()
        default_model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.executable if getattr(sys, 'frozen', False) else __file__))))), "stdnn_models", "models_cv_LR_S1")
        self.input_box1.setText(default_model_dir)
        row1_layout.addWidget(self.input_box1)
        self.button1 = QPushButton('浏览')
        row1_layout.addWidget(self.button1)

        # 第二行 提取数据获取
        row2_widget = QWidget() 
        row2_layout = QHBoxLayout(row2_widget)
        row2_layout.addWidget(QLabel('数据目录:'))
        self.input_box2 = QLineEdit()
        self.input_box2.setPlaceholderText('请输入路径...')
        row2_layout.addWidget(self.input_box2)
        self.button2 = QPushButton('浏览')
        row2_layout.addWidget(self.button2)

        # 第三行 保存地址
        row3_widget = QWidget() 
        row3_layout = QHBoxLayout(row3_widget)
        row3_layout.addWidget(QLabel('输出目录:'))
        self.input_box3 = QLineEdit()
        self.input_box3.setPlaceholderText('请输入路径...')
        row3_layout.addWidget(self.input_box3)
        self.button3 = QPushButton('浏览')
        row3_layout.addWidget(self.button3)

        # 第四行 生成按钮
        row4_widget = QWidget() 
        row4_layout = QHBoxLayout(row4_widget) 
        row4_layout.addSpacing(20) 
        row4_layout.addWidget(QLabel('并行提取样本数量:'))
        self.batch_input = QLineEdit()
        self.batch_input.setText('4')
        self.batch_input.setFixedWidth(40)
        row4_layout.addWidget(self.batch_input)
        row4_layout.addStretch()  # 弹簧占据左侧空间
        self.fingerprint_button = QPushButton('生成指纹')
        row4_layout.addWidget(self.fingerprint_button)
        # self.stop_button = QPushButton('停止')  
        # row4_layout.addWidget(self.stop_button)  
        row4_layout.setContentsMargins(0, 10, 100, 0)  

        # 第五行：终端风格的文本框
        row5_widget = QWidget()
        row5_layout = QVBoxLayout(row5_widget)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText('运行日志将显示在这里...')
        self.log_text.setStyleSheet("""
            QTextEdit, QPlainTextEdit {
                font-size: 14px;           /* 字体大小，可根据需要调整 */
                font-weight: bold;         /* 字体加粗变深 */
                color: #2c3e50;            /* 字体颜色，更深更明显 */
                font-family: 'Microsoft YaHei', 'SimHei', monospace;  /* 字体族 */
            }
        """)
        row5_layout.addWidget(self.log_text)

        # 添加到垂直布局
        layout.addWidget(row1_widget)
        layout.addWidget(row2_widget)
        layout.addWidget(row3_widget)
        layout.addWidget(row4_widget)
        layout.addWidget(row5_widget)

        # 添加弹簧把内容推上去
        layout.addStretch()  # 把两行推到顶部
    
    def initEvents(self):
        """初始化信号/事件绑定"""
        self.button1.clicked.connect(self.browse_folder1)
        self.button2.clicked.connect(self.browse_folder2)
        self.button3.clicked.connect(self.browse_folder3) 
        self.fingerprint_button.clicked.connect(self.generate_fingerprint) 
        # self.stop_button.clicked.connect(self.stop_generation) 
        
    def browse_folder1(self):
        """浏览文件夹"""
        current_dir = get_app_dir()  # 获取exe/.py所在目录
        parent_dir = os.path.dirname(current_dir)  # 上一层目录 
        parent_dir = os.path.dirname(parent_dir) 
        parent_dir = os.path.dirname(parent_dir) 
        default_dir = os.path.join(parent_dir, "stdnn_models")
        folder_path = QFileDialog.getExistingDirectory(self, '选择文件夹', default_dir)
        if folder_path:
            self.input_box1.setText(folder_path)

    def browse_folder2(self):
        """浏览文件夹"""
        current_dir = get_app_dir()  # 获取exe/.py所在目录
        parent_dir = os.path.dirname(current_dir)  # 上一层目录 
        parent_dir = os.path.dirname(parent_dir) 
        parent_dir = os.path.dirname(parent_dir) 
        folder_path = QFileDialog.getExistingDirectory(self, '选择文件夹',parent_dir)
        if folder_path:
            self.input_box2.setText(folder_path)

    def browse_folder3(self):
        """浏览文件夹"""
        current_dir = get_app_dir()  # 获取exe/.py所在目录
        parent_dir = os.path.dirname(current_dir)  # 上一层目录 
        parent_dir = os.path.dirname(parent_dir) 
        parent_dir = os.path.dirname(parent_dir) 
        folder_path = QFileDialog.getExistingDirectory(self, '选择文件夹',parent_dir)
        if folder_path:
            self.input_box3.setText(folder_path)
    
    def generate_fingerprint(self):
        """生成指纹  - 在后台线程中运行"""
        # print(f"DEBUG: generate_fingerprint 调用, 当前 worker = {self.worker}")
        global _stop_flag  
        _stop_flag = False  # 重置停止标志
        model_path = self.input_box1.text()  # 模型路径
        data_path = self.input_box2.text()   # 提取数据路径
        output_path = self.input_box3.text()   # 保存地址

        if not data_path or not output_path or not model_path:
            self.log_text.append("❌ 请先选择模型路径和提取数据路径")
            return
        
        # ✅ 检查并清理旧线程
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        
        # ✅ 清空文本框（先清空再添加日志）
        self.log_text.clear()
        
        # ✅ 禁用生成按钮，启用停止按钮
        self.fingerprint_button.setEnabled(False)
        # self.stop_button.setEnabled(True)
        
        try:
            batch = int(self.batch_input.text())
        except ValueError:
            batch = 4
        # 创建工作线程
        self.worker = WorkerThread(model_path,data_path, output_path, batch)
        
        # 添加调试：打印所有连接        
        self.worker.error.connect(self.on_generation_error)
        
        self.worker.finished.connect(self.on_generation_finished)

        
        self.worker.start()

    def stop_generation(self):
        """强行终止（用户主动点击停止）"""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            # self.stop_button.setEnabled(False)  # ✅ 新增：点击停止后立即禁用
            self.log_text.append("\n⚠ 正在停止... ")

    def on_generation_error(self, error_msg):
        """生成出错（异常结束）"""
        self.fingerprint_button.setEnabled(True)
        # self.stop_button.setEnabled(False)  # ✅ 正确：出错后也不需要停止了
        self.log_text.append(f"\n❌ 错误: {error_msg}")
        self.cleanup_worker()

    def on_generation_finished(self):
        """线程结束（正常完成或异常都会触发）"""
        self.fingerprint_button.setEnabled(True)
        self.cleanup_worker()

    def cleanup_worker(self):
        """清理工作线程"""
        if self.worker is not None:
            if self.worker.isRunning():
                self.worker.wait(500)  # 等待最多0.5秒
            self.worker.deleteLater()
            self.worker = None

if __name__ == '__main__':

    # 运行
    app = QApplication(sys.argv)
    print("2. QApplication创建完成:", time.time())
    window = MyWindow()
    print("3. 窗口实例创建完成:", time.time())
    window.show()
    print("4. 窗口显示完成:", time.time())
    sys.exit(app.exec())