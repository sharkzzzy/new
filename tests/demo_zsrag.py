import os
import torch
from zsrag.pipelines.pipeline_zsrag_inpaint import ZSRAGPipeline, run_zsrag

def main():
    # 基本参数
    width, height = 1024, 1024
    seed = 42
    save_dir = "outputs_zsrag_demo"
    os.makedirs(save_dir, exist_ok=True)

    global_prompt = "a wide photo of a park scene, green grass and trees, blurred background, sunny lighting, photorealistic, cinematic"

    subjects = [
        {
            "name": "cat",
            "prompt": "a red cat sitting, full body, fluffy, sharp details, left side, clearly separated",
            "position": "left",
            "neg_target": "blue dog, blue fur",
            # "ref_image": "assets/red_fur_patch.png",  # 可选
        },
        {
            "name": "dog",
            "prompt": "a blue dog sitting, full body, sharp details, right side, clearly separated",
            "position": "right",
            "neg_target": "red cat, red fur",
            # "ref_image": "assets/blue_fur_patch.png",
        }
    ]

    # 1. 实例化 Pipeline
    print("Loading Pipeline...")
    pipe = ZSRAGPipeline(fp32_unet=True)

    # 2. 运行全流程
    print("Running ZS-RAG...")
    result = run_zsrag(
        pipeline=pipe,  # 传入实例
        global_prompt=global_prompt,
        subjects=subjects,
        width=width,
        height=height,
        seed=seed,
        bootstrap_steps=20,
        bootstrap_guidance=5.5,
        safety_expand=0.05,
        tau=0.8,
        bg_floor=0.05,
        gap_ratio=0.08,
        num_inference_steps=40,
        cfg_pos=7.5,
        cfg_neg=3.0,
        kappa=1.0,
        iter_clip=2,
        clip_threshold=0.32,
        probe_interval=5,
        sde_strength=0.3,
        sde_steps=20,
        sde_guidance=5.0,
        save_dir=save_dir,
    )

    print(f"Done. Results saved in {save_dir}")
    # result["bootstrap"]["image"] -> 初始图（PIL）
    # result["bind"]["image"] -> 绑定后图像张量 [1,3,H,W]
    # result["harmonized"] -> SDEdit 重整后图像（PIL）

if __name__ == "__main__":
    main()
