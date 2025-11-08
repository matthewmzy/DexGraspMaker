# main.py (moved from interactive_gripper_tool)
import sys
import argparse
from PyQt6.QtWidgets import QApplication

# Support running as "python scripts/main.py" and "python -m scripts.main"
try:
    from scripts.main_window import MainWindow  # when project root is on sys.path
except ModuleNotFoundError:
    import os as _os, sys as _sys
    _sys.path.append(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from scripts.main_window import MainWindow


def main():
    parser = argparse.ArgumentParser(description="DexGraspMaker Interactive Tool")
    parser.add_argument('-d', '--load-default', action='store_true', help='加载默认测试资源')
    parser.add_argument('-hd', '--hand', dest='hand', default='shadow', help='选择手配置名称 (默认: shadow)')
    args, unknown = parser.parse_known_args()
    load_default = args.load_default
    selected_hand = args.hand

    app = QApplication(sys.argv)
    try:
        window = MainWindow(load_default=load_default, selected_hand=selected_hand)
        window.setWindowTitle("可交互式机械手位姿匹配工具 (v0.1)")
        window.setGeometry(100, 100, 1600, 900)
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"启动应用程序时发生严重错误: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
