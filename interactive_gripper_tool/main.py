# main.py

import sys
import argparse
from PyQt6.QtWidgets import QApplication

from main_window import MainWindow


def main():
    """
    应用程序的主入口函数。
    """
    # 命令行参数
    parser = argparse.ArgumentParser(description="DexGraspMaker Interactive Tool")
    parser.add_argument('-d', '--load-default', action='store_true', help='加载默认测试资源')
    parser.add_argument('-hd', '--hand', dest='hand', default='shadow', help='选择手配置名称 (默认: shadow)')
    args, unknown = parser.parse_known_args()
    load_default = args.load_default
    selected_hand = args.hand
    
    # 1. 创建 QApplication 实例
    # sys.argv 允许从命令行传递参数给应用程序
    app = QApplication(sys.argv)

    # 2. 创建 MainWindow 实例
    # 这是我们应用程序的主窗口
    try:
        window = MainWindow(load_default=load_default, selected_hand=selected_hand)
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