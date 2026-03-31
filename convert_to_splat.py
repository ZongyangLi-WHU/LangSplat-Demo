import numpy as np
from plyfile import PlyData
import struct

def convert_ply_to_splat(input_ply, output_splat, max_points=500000): # 限制 50 万点以内，保证不卡
    print(f"读取并裁剪: {input_ply} ...")
    plydata = PlyData.read(input_ply)
    v = plydata['vertex']
    
    # 裁剪
    mask = (
        (v['x'] >= -2.47) & (v['x'] <= 4.02) &
        (v['y'] >= -4.62) & (v['y'] <= 5.32) &
        (v['z'] >= -2.02) & (v['z'] <= 5.44)
    )
    
    x, y, z = v['x'][mask], v['y'][mask], v['z'][mask]
    f_dc_0, f_dc_1, f_dc_2 = v['f_dc_0'][mask], v['f_dc_1'][mask], v['f_dc_2'][mask]
    opacity = v['opacity'][mask]
    scale_0, scale_1, scale_2 = v['scale_0'][mask], v['scale_1'][mask], v['scale_2'][mask]
    rot_0, rot_1, rot_2, rot_3 = v['rot_0'][mask], v['rot_1'][mask], v['rot_2'][mask], v['rot_3'][mask]

    # 🔥 终极除垢滤网：剔除所有 NaN、无穷大等会让显卡崩溃的“毒粒子”
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(scale_0)
    x, y, z = x[valid], y[valid], z[valid]
    f_dc_0, f_dc_1, f_dc_2 = f_dc_0[valid], f_dc_1[valid], f_dc_2[valid]
    opacity, scale_0, scale_1, scale_2 = opacity[valid], scale_0[valid], scale_1[valid], scale_2[valid]
    rot_0, rot_1, rot_2, rot_3 = rot_0[valid], rot_1[valid], rot_2[valid], rot_3[valid]

    total_points = len(x)
    if total_points > max_points:
        indices = np.random.choice(total_points, max_points, replace=False)
        x, y, z = x[indices], y[indices], z[indices]
        f_dc_0, f_dc_1, f_dc_2 = f_dc_0[indices], f_dc_1[indices], f_dc_2[indices]
        opacity = opacity[indices]
        scale_0, scale_1, scale_2 = scale_0[indices], scale_1[indices], scale_2[indices]
        rot_0, rot_1, rot_2, rot_3 = rot_0[indices], rot_1[indices], rot_2[indices], rot_3[indices]

    print("正在写入 WebGL 强制小端序 (Little-Endian) 格式...")
    with open(output_splat, 'wb') as f:
        for i in range(len(x)):
            # 🔥 注意看这里多了一个 '<' 符号，代表强制小端序！
            f.write(struct.pack('<fff', x[i], y[i], z[i]))
            
            s0 = np.exp(np.clip(scale_0[i], -20, 5))
            s1 = np.exp(np.clip(scale_1[i], -20, 5))
            s2 = np.exp(np.clip(scale_2[i], -20, 5))
            f.write(struct.pack('<fff', s0, s1, s2))
            
            r = np.clip((f_dc_0[i] * 0.28209 + 0.5) * 255, 0, 255)
            g = np.clip((f_dc_1[i] * 0.28209 + 0.5) * 255, 0, 255)
            b = np.clip((f_dc_2[i] * 0.28209 + 0.5) * 255, 0, 255)
            a = np.clip((1 / (1 + np.exp(-opacity[i]))) * 255, 0, 255)
            f.write(struct.pack('<BBBB', int(r), int(g), int(b), int(a)))
            
            q = np.array([rot_0[i], rot_1[i], rot_2[i], rot_3[i]])
            q_norm = np.linalg.norm(q)
            if q_norm == 0: q_norm = 1
            q = q / q_norm
            q = np.clip((q * 128 + 128), 0, 255).astype(np.uint8)
            f.write(struct.pack('<BBBB', q[0], q[1], q[2], q[3]))

    print(f"🎉 转换完成！快去上传吧！")

if __name__ == "__main__":
    convert_ply_to_splat("point_cloud.ply", "my_desk.splat")