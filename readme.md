

1. 数据准备
    FF++下载地址：https://github.com/ondyari/FaceForensics
    CelebDF下载地址：https://github.com/yuezunli/celeb-deepfakeforensics

2. 数据预处理：
    2.1 使用CenterFace[1]来检测人脸区域，再将提取的人脸图像分辨率缩放到299×299，并保存为png格式。
    2.2 利用SSIM[2]来计算真伪图像的结构差异，并按本文的制作方法来获得SSIM mask，获得的SSIM掩码图即基准分割图。

3. 运行训练脚本：
   3.1 修改脚本中的FF++数据集的路径：
	PATH_DATA='/home/xxx/FFpp/SegDataV3'
	该路径包含FF++人脸区域的视频帧和其伪造区域的SSIM分割图。
   3.2 目录结构：
├── checkpoints		   模型权重保存路径
├── readme.txt	           说明文档
├── run.sh	           训练运行脚本
├── seg	                   模型配置和训练相关代码
│   ├── cfgs               配置文件
│   ├── libs               辅助文件
│   ├── modules            模型相关的文件
│   ├── trainG9.3.py       训练主脚本
└── tmp_cfg                缓存配置文件

   3.3 运行训练脚本：
	./run.sh



[1] Y. Xu, W. Yan, G. Yang, J. Luo, T. Li, and J. He, “CenterFace: Joint face detection and alignment using face as point,” Sci. Program., vol. 2020, pp. 1–8, Jul. 2020.
[2] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: From error visibility to structural similarity,” IEEE Trans. Image Process., vol. 13, no. 4, pp. 600–612, Apr. 2004.