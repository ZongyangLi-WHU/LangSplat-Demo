import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
import sys
sys.path.append(".")
from autoencoder.model import Autoencoder
from eval.openclip_encoder import OpenCLIPNetwork
import eval.colormaps as colormaps

# 1. 跨模态查询配置
PROMPTS = ["a Canon camera lens", "a plush toy", "a hand gripper", "a book", "a power bank"]
DATASET_NAME = "my_desk"
FEATURE_NPY_DIR = f"./output/{DATASET_NAME}_langsplat_1/train/ours_30000/renders_npy"
RGB_IMG_DIR = f"./output/{DATASET_NAME}_langsplat_1/train/ours_30000/gt"
AE_CKPT_PATH = f"./autoencoder/ckpt/{DATASET_NAME}/best_ckpt.pth"
OUTPUT_DIR = f"./output/{DATASET_NAME}_highlight_frames"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("加载 CLIP 大模型和 Autoencoder...")
    clip_model = OpenCLIPNetwork(device)
    clip_model.set_positives(PROMPTS)
    
    # 实例化自编码器并加载降维权重
    model = Autoencoder([256, 128, 64, 32, 3], [16, 32, 64, 128, 256, 256, 512]).to(device)
    model.load_state_dict(torch.load(AE_CKPT_PATH, map_location=device))
    model.eval()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for prompt in PROMPTS:
        os.makedirs(os.path.join(OUTPUT_DIR, prompt), exist_ok=True)
        
    npy_files = sorted([f for f in os.listdir(FEATURE_NPY_DIR) if f.endswith('.npy')])
    print("开始生成文本高亮热力图...")
    
    for filename in tqdm(npy_files):
        # 2. 加载三维特征与 RGB 真彩图像
        feat_path = os.path.join(FEATURE_NPY_DIR, filename)
        img_path = os.path.join(RGB_IMG_DIR, filename.replace('.npy', '.png'))
        
        sem_feat = torch.from_numpy(np.load(feat_path)).float().to(device)
        rgb_img = cv2.imread(img_path)[..., ::-1] # BGR to RGB
        rgb_img = torch.from_numpy((rgb_img/255.0).astype(np.float32)).to(device)
        
        # 3. 语义解码：通过自编码器将 3 维特征膨胀回 512 维
        with torch.no_grad():
            h, w, _ = sem_feat.shape
            restored_feat = model.decode(sem_feat.flatten(0, 1)).view(1, h, w, -1)
            
            # 4. 计算文本与图像特征在 512 维空间中的余弦相似度
            valid_map = clip_model.get_max_across(restored_feat) 
            
            # 5. 生成物理对象高亮叠加图
            for k, prompt in enumerate(PROMPTS):
                relevancy_map = valid_map[0, k]
                np_relev = relevancy_map.cpu().numpy()
                
                # 图像平滑与特征归一化
                kernel = np.ones((30, 30)) / 900.0
                avg_filtered = torch.from_numpy(cv2.filter2D(np_relev, -1, kernel)).to(device)
                torch_relev = 0.5 * (avg_filtered + relevancy_map)
                
                # 设定置信度阈值 (0.5)，低于该阈值的非目标区域予以压暗处理
                p_i = torch.clip(torch_relev - 0.5, 0, 1).unsqueeze(-1)
                valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
                
                mask = (torch_relev < 0.5).squeeze()
                valid_composited[mask, :] = rgb_img[mask, :] * 0.3 # 背景亮度压暗至 30%
                
                # 图像落盘与转换
                save_path = os.path.join(OUTPUT_DIR, prompt, filename.replace('.npy', '.jpg'))
                output_image = (valid_composited.cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(save_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    main()
