import os
import torch
import cv2
import numpy as np
import glob
from torchvision import transforms
from PIL import Image
import sys

# 🚨 导入核心模型 (确保 core/net.py 是最新版)
from core.net import MultiBranchNet


def get_tiny_config():
    return {
        'dataset': 'tiny_imagenet',
        'backbone': 'resnet18',
        'img_size': 64,  # 模型输入固定为 64
        'gate_temp': 1.0,
        'num_known': 20,
        'batch_size': 1,
        'gpu_ids': '0'
    }


def find_latest_checkpoint(root_dir='./ckpt/osr/tiny_imagenet'):
    """自动寻找最新的权重文件"""
    if not os.path.exists(root_dir): return None
    subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    if not subdirs: return None
    latest_subdir = max(subdirs, key=os.path.getmtime)

    # 优先找 model_best.pth
    best_path = os.path.join(latest_subdir, 'model_best.pth')
    if os.path.exists(best_path): return best_path

    # 其次找 best_model.pth
    best_path_alt = os.path.join(latest_subdir, 'best_model.pth')
    if os.path.exists(best_path_alt): return best_path_alt
    return None


def preprocess_image(img_path, model_img_size=64):
    """
    分离处理：
    1. input_tensor: 缩放到 64x64 给模型推理
    2. vis_base: 保持原图高清尺寸用于展示
    """
    try:
        raw_img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Skipping bad image {img_path}: {e}")
        return None, None

    # 1. 制作模型输入
    transform = transforms.Compose([
        transforms.Resize((model_img_size, model_img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(raw_img).unsqueeze(0)

    # 2. 制作可视化底图 (保持原汁原味)
    vis_img = np.array(raw_img)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

    return input_tensor, vis_img


def visualize_cam_on_image(img, cam):
    # 1. 暴力放大: 将 8x8 的 CAM 插值放大到原图尺寸 (例如 800x800)
    # 使用 INTER_CUBIC 保证热力图圆润平滑
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    # 2. 生成热力图
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    # 3. 叠加: 原图(0.6) + 热力图(0.4)
    # 这样既能看清物体纹理，又能看清热力分布
    cam_result = heatmap * 0.4 + np.float32(img) / 255 * 0.6

    # 4. 归一化防过曝
    cam_result = cam_result / np.max(cam_result)
    return np.uint8(255 * cam_result)


def main():
    options = get_tiny_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🏗️  Building MultiBranchNet (Vis_CAM Mode)...")
    model = MultiBranchNet(options).to(device)

    # 自动加载权重
    ckpt_path = find_latest_checkpoint()
    if ckpt_path is None:
        print("❌ No checkpoint found in ./ckpt/osr/tiny_imagenet")
        return

    print(f"📥 Loading weights: {ckpt_path}")
    checkpoint = torch.load(ckpt_path)
    state_dict = checkpoint['state_dict']
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("✅ Weights loaded successfully (Strict).")
    except:
        model.load_state_dict(new_state_dict, strict=False)
        print("⚠️ Weights loaded with strict=False.")

    model.eval()

    # 读取图片
    img_files = glob.glob("./demo_images/*.jpg")
    if len(img_files) == 0:
        print("❌ No images in ./demo_images/. Run get_demo_images.py first.")
        return

    print(f"🚀 Processing {len(img_files)} images...")

    # 结果保存目录
    save_dir = "./vis_cam_results"
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    for img_file in img_files:
        filename = os.path.basename(img_file)

        # 获取高清底图(vis_base) 和 模型输入(input_tensor)
        input_tensor, vis_base = preprocess_image(img_file, model_img_size=64)
        if input_tensor is None: continue

        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            cams = outputs['cams']

            # 取 Branch 1 的 CAM (最稳定的主分支)
            cam = cams[0, 0, :, :].cpu().numpy()

            # 归一化 CAM (去除底噪，让红的地方更红)
            cam = cam - np.min(cam)
            cam = cam / (np.max(cam) + 1e-8)

            # 生成叠加图
            vis_cam = visualize_cam_on_image(vis_base, cam)

            # 拼接: [原图] | [热力图]
            # 纯净版：没有任何文字标签
            concat_img = np.hstack([vis_base, vis_cam])

            save_path = os.path.join(save_dir, "CAM_" + filename)
            cv2.imwrite(save_path, concat_img)
            print(f"   ✨ Saved: {save_path}")

    print(f"\n🎉 Visualization finished! Check folder: {save_dir}")


if __name__ == "__main__":
    main()