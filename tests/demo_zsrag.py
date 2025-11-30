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
        
        # Stage B (你的参数建议)
        safety_expand=0.0,
        tau=1.0,             # 互斥更弱
        bg_floor=0.18,       # 背景底权提高
        gap_ratio=0.12,      # 中缝加宽
        
        # Stage C
        num_inference_steps=40,
        cfg_pos=7.5,
        cfg_neg=3.0,
        kappa=1.0,           # 保持 1.0 即可，因为逻辑改了
        iter_clip=1,         # 跑一轮迭代即可验证
        clip_threshold=0.32,
        probe_interval=5,
        
        # Stage D
        sde_strength=0.3,
        sde_steps=20,
        sde_guidance=5.0,
        
        save_dir=save_dir,
    )

    print(f"Done. Results saved in {save_dir}")

if __name__ == "__main__":
    main()
