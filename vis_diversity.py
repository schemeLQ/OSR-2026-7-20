import os
import torch
import cv2
import numpy as np
import glob
from torchvision import transforms
from PIL import Image
import sys

# 🚨 导入你的核心模型
from core.net import MultiBranchNet


def get_tiny_config():
    # 必须与训练时的配置一致
    return {
        'dataset': 'tiny_imagenet',
        'backbone': 'resnet18',
        'img_size': 64,
        'gate_temp': 1.0,
        'num_known': 20,
        'batch_size': 1,
        'gpu_ids': '0'
    }


def find_latest_checkpoint(root_dir='./ckpt/osr/tiny_imagenet'):
    """
    自动寻找最近一次训练生成的最佳模型
    """
    if not os.path.exists(root_dir):
        print(f"❌ Error: Checkpoint root directory not found: {root_dir}")
        return None

    # 找到所有时间戳文件夹
    subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    if not subdirs:
        print(f"❌ Error: No timestamp folders found in {root_dir}")
        return None

    # 按修改时间排序，找最新的
    latest_subdir = max(subdirs, key=os.path.getmtime)

    # 优先找 model_best.pth
    best_path = os.path.join(latest_subdir, 'model_best.pth')
    if os.path.exists(best_path):
        return best_path

    # 其次找 best_model.pth
    best_path_alt = os.path.join(latest_subdir, 'best_model.pth')
    if os.path.exists(best_path_alt):
        return best_path_alt

    print(f"❌ Error: No 'model_best.pth' found in {latest_subdir}")
    return None


def preprocess_image(img_path, img_size=64):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    try:
        raw_img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Skipping bad image {img_path}: {e}")
        return None, None

    input_tensor = transform(raw_img).unsqueeze(0)

    # 保留底图用于叠加 (Resize 到 64x64 以匹配特征图)
    vis_img = np.array(raw_img.resize((img_size, img_size)))
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

    return input_tensor, vis_img


def visualize_cam_on_image(img, cam):
    # 🚨 [关键修复] 先把 8x8 的 CAM 放大到 64x64 (和原图 img 一样大)
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))

    # 热力图生成
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    # 叠加：原图 0.5 + 热力图 0.5
    cam_result = heatmap + np.float32(img) / 255
    cam_result = cam_result / np.max(cam_result)
    return np.uint8(255 * cam_result)


def main():
    options = get_tiny_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🏗️ Building MultiBranchNet (with VisualBACL)...")
    model = MultiBranchNet(options).to(device)

    # --- 自动寻找权重 ---
    ckpt_path = find_latest_checkpoint()
    if ckpt_path is None:
        return

    print(f"📥 Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path)

    # 处理 module. 前缀
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v

    # 尝试加载
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("✅ Success! Model loaded perfectly (Strict Mode).")
    except Exception as e:
        print(f"⚠️ Strict loading failed: {e}")
        print("Trying strict=False...")
        model.load_state_dict(new_state_dict, strict=False)

    model.eval()

    # --- 寻找图片 ---
    img_files = glob.glob("./demo_images/*.jpg")
    if len(img_files) == 0:
        print("❌ No images found in ./demo_images/ folder.")
        print("👉 Please run 'get_demo_images.py' first.")
        return

    print(f"🚀 Found {len(img_files)} images. Inferencing...")

    # 保存路径
    save_dir = "./vis_diversity"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"📂 Created directory: {save_dir}")

    for img_file in img_files:
        filename = os.path.basename(img_file)

        input_tensor, vis_base = preprocess_image(img_file)
        if input_tensor is None: continue

        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            cams = outputs['cams']
            gate_pred = outputs['gate_pred']  # [1, 2]

            # 假设 index 1 是已知类分数
            known_score = gate_pred[0, 1].item()

            # 使用 Branch 1 的 CAM
            cam = cams[0, 0, :, :].cpu().numpy()

            # 归一化
            cam = cam - np.min(cam)
            cam = cam / (np.max(cam) + 1e-8)

            # 画图 (现在会自动 resize 了)
            vis_cam = visualize_cam_on_image(vis_base, cam)

            # 在图上写分数
            label_text = f"Known: {known_score:.2f}"

            # 拼接：左边原图，右边热力图
            concat_img = np.hstack([vis_base, vis_cam])

            # 加上文字标签
            cv2.putText(concat_img, label_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            save_path = os.path.join(save_dir, "res_" + filename)
            cv2.imwrite(save_path, concat_img)
            print(f"   💾 Saved to {save_dir}: {filename} | {label_text}")

    print(f"\n🎉 Done! All results saved to {save_dir}")


if __name__ == "__main__":
    main()