controlnet_utils.py # ControlNet/T2I-Adapter 接入（线稿/深度）
sam_seg.py # SAM 分割；支持 Grounded-DINO 的框输入
grounded_dino_utils.py # 文本检测得到主体框
depth_lineart_extract.py # 从初始图提取深度与线稿（ZoeDepth/MiDaS + Canny/Lineart）
scene_planner.py # 场景图解析与布局（可选：LLM 支持）
utils.py # 通用工具（seed、图像IO、颜色度量等）
pipelines/
pipeline_zsrag_inpaint.py # 主 Pipeline：三阶段流程（几何自举→属性绑定→全局重整）
runner.py # 命令行入口，加载配置与执行
configs/
defaults.yaml # 默认参数配置（CFG/门控层/互斥温度等）
tests/
demo_zsrag.py # 演示脚本：零训练、两主体属性绑定示例
