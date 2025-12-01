import os
import torch


from zsrag.pipelines.pipeline_zsrag_inpaint import ZSRAGPipeline, run_zsrag

def main():
    width, height = 768, 768 # 降分辨率保显存
    seed = 42
    save_dir = "outputs_zsrag_fix_final"
    os.makedirs(save_dir, exist_ok=True)

    global_prompt = "a wide photo of a park scene, green grass and trees, blurred background, sunny lighting, photorealistic, cinematic, 8k"

    subjects = [
        {
            "name": "cat",
            "prompt": "a red cat sitting, full body, fluffy, sharp details, left side, clearly separated",
            "position": "left",
            "neg_target": "blue dog, blue fur",
            "ip_weight": 0.8, # 初始权重调低
        },
        {
            "name": "dog",
            "prompt": "a blue dog sitting, full body, sharp details, right side, clearly separated",
            "position": "right",
            "neg_target": "red cat, red fur",
            "ip_weight": 0.8,
        }
    ]

    print("Loading Pipeline...")
    pipe = ZSRAGPipeline(fp32_unet=True)

    print("Running ZS-RAG (Final Logic)...")
    result = run_zsrag(
        pipeline=pipe,
        global_prompt=global_prompt,
        subjects=subjects,
        width=width,
        height=height,
        seed=seed,
        
        # Stage A
        bootstrap_steps=20,
        bootstrap_guidance=5.5,
        safety_expand=0.0,
        tau=1.0,
        bg_floor=0.18,
        gap_ratio=0.12,
        feather_radius_img=7,
        feather_radius_lat=1,
        
        # Stage C (强力生成主体)
        kappa=1.5,           # 强力注入
        start_strength=0.90, # 几乎重绘
        delay_gate_step=8,   # 延迟门控
        iter_clip=1,         # 跑一轮即可
        
        # Stage D (参数已在 run_zsrag 内部 Two-Pass 逻辑中定死，这里传参主要影响默认值)
        sde_strength=0.35, 
        sde_steps=20,
        sde_guidance=5.0,
        
        save_dir=save_dir,
    )

    print(f"Done. Results saved in {save_dir}")

if __name__ == "__main__":
    main()
