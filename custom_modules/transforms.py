import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS

@TRANSFORMS.register_module()
class ThermalNoiseInjection(BaseTransform):
    """针对热红外极性归一化图像的特定噪声注入。"""
    
    def __init__(self, noise_level: float = 0.05):
        self.noise_level = noise_level

    def transform(self, results: dict) -> dict:
        """纯函数设计，只修改并返回字典，不破坏原有张量结构。
        加入微弱高斯噪声以应对无纹理特点，遵循禁止任意反转极性的领域约束。
        """
        if 'img' not in results:
            return results
            
        img = results['img']
        # 基于图像标准差或者指定幅度注入噪声
        noise = np.random.normal(0, 255 * self.noise_level, img.shape)
        # 热红外图中高温区域更亮，所以我们加上噪声后硬截断 (clip) 并保持 dtype
        results['img'] = np.clip(img + noise, 0, 255).astype(img.dtype)
        return results
