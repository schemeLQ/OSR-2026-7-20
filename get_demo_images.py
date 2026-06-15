import os
import requests
import urllib3
import time
import numpy as np
import cv2

# 忽略 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

save_dir = './demo_images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# === 25 张高清测试图 (Picsum ID 映射) ===
# 覆盖：动物、植物、交通、建筑、日常用品、自然景观
images = {
    # --- 1. 动物类 (Animals) ---
    "dog_black.jpg": "https://picsum.photos/id/237/800/800",      # 黑狗
    "dog_pug.jpg": "https://picsum.photos/id/1025/800/800",       # 八哥犬 (被子包裹)
    "lioness.jpg": "https://picsum.photos/id/1074/800/800",       # 狮子/大型猫科
    "deer.jpg": "https://picsum.photos/id/1003/800/800",          # 鹿
    "bird_sea.jpg": "https://picsum.photos/id/1024/800/800",      # 鸟 (猛禽/海鸟)
    "bear_wild.jpg": "https://picsum.photos/id/1020/800/800",     # 熊
    "insect_macro.jpg": "https://picsum.photos/id/115/800/800",   # 昆虫/草地
    "fish_sealife.jpg": "https://picsum.photos/id/1069/800/800",  # 海洋生物(水母/鱼类)

    # --- 2. 交通工具 (Vehicles) ---
    "car_vintage.jpg": "https://picsum.photos/id/111/800/800",    # 老爷车
    "car_traffic.jpg": "https://picsum.photos/id/183/800/800",    # 现代车/巴士
    "train_rail.jpg": "https://picsum.photos/id/1026/800/800",    # 火车/铁轨
    "boat_sea.jpg": "https://picsum.photos/id/211/800/800",       # 船/游艇
    "bicycle.jpg": "https://picsum.photos/id/146/800/800",        # 自行车/推车 (或相关结构)

    # --- 3. 建筑与结构 (Architecture) ---
    "castle.jpg": "https://picsum.photos/id/1040/800/800",        # 城堡
    "lighthouse.jpg": "https://picsum.photos/id/870/800/800",     # 灯塔
    "bridge_suspension.jpg": "https://picsum.photos/id/122/800/800", # 吊桥
    "city_building.jpg": "https://picsum.photos/id/195/800/800",  # 城市建筑

    # --- 4. 植物与自然 (Nature) ---
    "flower_macro.jpg": "https://picsum.photos/id/152/800/800",   # 花朵
    "forest_tree.jpg": "https://picsum.photos/id/28/800/800",     # 森林/树木
    "mountain_snow.jpg": "https://picsum.photos/id/29/800/800",   # 雪山
    "beach_sand.jpg": "https://picsum.photos/id/100/800/800",     # 沙滩/海岸

    # --- 5. 日常物品 (Objects/Indoor) ---
    "computer_work.jpg": "https://picsum.photos/id/0/800/800",    # 电脑/办公桌
    "book_library.jpg": "https://picsum.photos/id/24/800/800",    # 书籍
    "coffee_cup.jpg": "https://picsum.photos/id/30/800/800",      # 杯子
    "camera_lens.jpg": "https://picsum.photos/id/250/800/800",    # 相机/精密仪器
}

print(f"🚀 准备下载 {len(images)} 张高清多样化样张 (Picsum 源)...")
headers = {'User-Agent': 'Mozilla/5.0'}

# === 备用方案：万一断网，自动画图 ===
def create_fallback_image(filename):
    print(f"   ⚠️ 网络不通，正在本地生成测试图: {filename}")
    img = np.zeros((800, 800, 3), dtype=np.uint8)
    # 画个渐变色
    for i in range(800):
        img[i, :, 0] = (i * 255 // 800)
        img[:, i, 1] = ((800 - i) * 255 // 800)
    # 画个圈和文字
    cv2.circle(img, (400, 400), 150, (0, 0, 255), -1)
    cv2.putText(img, filename.split('.')[0], (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(save_dir, filename), img)
    print(f"   ✅ 本地生成成功！")

# === 主下载逻辑 ===
for name, url in images.items():
    save_path = os.path.join(save_dir, name)

    if os.path.exists(save_path) and os.path.getsize(save_path) > 1024:
        print(f"⏩ 跳过 (已存在): {name}")
        continue

    try:
        print(f"⬇️ 正在下载: {name} ...")
        # timeout=10秒，防止卡死
        response = requests.get(url, headers=headers, stream=True, verify=False, timeout=15)

        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"   ✅ 下载成功!")
        else:
            print(f"   ❌ 服务器返回 {response.status_code}，切换备用...")
            create_fallback_image(name)

    except Exception as e:
        print(f"   ❌ 网络报错: {e} -> 切换备用")
        create_fallback_image(name)

print(f"\n🎉 25张图像准备完毕！文件夹: {save_dir}")
print("👉 下一步：运行你的可视化脚本 (vis_diversity.py 等) 来查看模型对这些类别的反应。")