# main.py

import sys
from PyQt6.QtWidgets import QApplication

# 导入我们（尚未实现的）主窗口类
# 我们假设 'main_window.py' 文件中会定义一个名为 'MainWindow' 的类
try:
    from main_window import MainWindow
except ImportError:
    print("错误：无法从 'main_window.py' 导入 'MainWindow'。")
    print("请确保该文件存在并且类名正确。")
    # 暂时创建一个占位符类以便程序能运行（仅用于演示）
    if 'MainWindow' not in globals():
        from PyQt6.QtWidgets import QMainWindow, QLabel
        class MainWindow(QMainWindow):
            def __init__(self):
                super().__init__()
                self.setCentralWidget(QLabel("占位符：MainWindow 未实现"))
                self.resize(800, 600)

def main():
    """
    应用程序的主入口函数。
    """
    
    # 1. 创建 QApplication 实例
    # sys.argv 允许从命令行传递参数给应用程序
    app = QApplication(sys.argv)

    # 2. 创建 MainWindow 实例
    # 这是我们应用程序的主窗口
    try:
        window = MainWindow()
        window.setWindowTitle("可交互式机械手位姿匹配工具 (v0.1)")
        
        # 设置一个合理的初始大小
        window.setGeometry(100, 100, 1600, 900) 

        # 3. 显示主窗口
        window.show()

        # 4. 执行应用程序的事件循环
        # app.exec() 会启动Qt的事件处理机制，程序将在此处阻塞，
        # 直到应用程序退出（例如关闭主窗口）。
        sys.exit(app.exec())

    except Exception as e:
        print(f"启动应用程序时发生严重错误: {e}")
        # 在这里可以添加一个QMessageBox来显示错误
        sys.exit(1)


if __name__ == '__main__':
    # 确保Python脚本作为主程序运行时才执行main()
    main()