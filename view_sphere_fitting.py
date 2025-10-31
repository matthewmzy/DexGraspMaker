#!/usr/bin/env python3
"""
查看Shadow手球体拟合可视化结果
"""

import os
import sys
from PIL import Image
import matplotlib.pyplot as plt

def show_sphere_fitting_results():
    """显示球体拟合可视化结果"""

    # 查找所有球体拟合图像
    image_files = [f for f in os.listdir('.') if f.startswith('sphere_fitting_') and f.endswith('.png')]

    if not image_files:
        print("没有找到球体拟合可视化图像文件")
        return

    print(f"找到 {len(image_files)} 个球体拟合可视化图像")
    print("\n可用的可视化:")

    # 按link名称排序
    image_files.sort()

    for i, filename in enumerate(image_files, 1):
        link_name = filename.replace('sphere_fitting_', '').replace('.png', '')
        print(f"{i:2d}. {link_name}")

    print("\n概览图像:")
    print("25. shadow_hand_sphere_fitting.png (所有links概览)")

    # 询问用户想看哪个
    while True:
        try:
            choice = input("\n请选择要查看的图像编号 (1-24为单个link, 25为概览, q退出): ").strip().lower()

            if choice == 'q':
                break

            choice_num = int(choice)

            if 1 <= choice_num <= 24:
                filename = image_files[choice_num - 1]
                link_name = filename.replace('sphere_fitting_', '').replace('.png', '')
                print(f"\n显示 {link_name} 的球体拟合结果...")

                # 使用PIL显示图像
                img = Image.open(filename)
                plt.figure(figsize=(12, 8))
                plt.imshow(img)
                plt.title(f"Sphere Fitting: {link_name}")
                plt.axis('off')
                plt.show()

            elif choice_num == 25:
                filename = "shadow_hand_sphere_fitting.png"
                if os.path.exists(filename):
                    print("\n显示所有links的球体拟合概览...")
                    img = Image.open(filename)
                    plt.figure(figsize=(15, 10))
                    plt.imshow(img)
                    plt.title("Shadow Hand Sphere Fitting Overview")
                    plt.axis('off')
                    plt.show()
                else:
                    print("概览图像不存在")

            else:
                print("无效选择，请输入1-25或q")

        except ValueError:
            print("请输入有效的数字或q")
        except KeyboardInterrupt:
            break

if __name__ == '__main__':
    show_sphere_fitting_results()