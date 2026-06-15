import os
import shutil

# 设置你的数据路径
val_dir = './data/tiny-imagenet-200/val'
annot_file = os.path.join(val_dir, 'val_annotations.txt')

if os.path.exists(annot_file):
    print("正在整理 Tiny-ImageNet 验证集...")
    with open(annot_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split('\t')
            img_file = parts[0]
            cls_id = parts[1]

            # 创建类别文件夹
            cls_dir = os.path.join(val_dir, cls_id)
            if not os.path.exists(cls_dir):
                os.makedirs(cls_dir)

            # 移动图片
            src_img = os.path.join(val_dir, 'images', img_file)
            dst_img = os.path.join(cls_dir, img_file)
            if os.path.exists(src_img):
                shutil.move(src_img, dst_img)
    print("✅ 验证集整理完成！")
else:
    print("❌ 未找到标注文件，请检查路径是否为 ./data/tiny-imagenet-200/val")