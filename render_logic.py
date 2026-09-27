import torch, cv2, numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    AutoPipelineForImage2Image,
    LCMScheduler,
    AutoencoderTiny
)

GW, GH = 512, 384

class NeuroRender:
    # Добавлен параметр use_taesd (по умолчанию True для обратной совместимости)
    def __init__(self, mode="turbo", compile_unet=False, remote_conn=None, use_taesd=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16
        self.mode = mode.lower().replace("_", "-")
        self.remote_conn = remote_conn
        self.use_taesd = use_taesd
        
        if self.remote_conn is not None:
            print(f"[NeuroRender] Active in REMOTE CLIENT mode ({self.mode.upper()}).")
            return
        
        print(f"[NeuroRender] Initializing {self.mode.upper()} pipeline on {self.device} (TAESD: {self.use_taesd})...")
        
        if self.mode in ["sdxl-turbo", "sdxl"]:
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                "stabilityai/sdxl-turbo",
                torch_dtype=self.dtype,
                variant="fp16"
            ).to(self.device)

        elif self.mode in ["turbo", "sd-turbo"]:
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                "stabilityai/sd-turbo",
                torch_dtype=self.dtype,
                variant="fp16"
            ).to(self.device)

        else:  # default: lcm
            self.mode = "lcm"
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "SimianLuo/LCM_Dreamshaper_v7",
                torch_dtype=self.dtype
            ).to(self.device)
            self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
            
            # Условие отключения TAESD
            if self.use_taesd:
                try:
                    self.pipe.vae = AutoencoderTiny.from_pretrained(
                        "madebyollin/taesd",
                        torch_dtype=self.dtype
                    ).to(self.device)
                except Exception as e:
                    print(f"[!] TAESD failed to load: {e}")
            else:
                print("[NeuroRender] TAESD bypassed. Using standard SD 1.5 VAE.")

        self.pipe.safety_checker = None
        self.pipe.set_progress_bar_config(disable=True)
        
        try:
            self.pipe.enable_attention_slicing()
        except Exception:
            pass
            
        if compile_unet and self.device == 'cuda':
            try:
                self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=False)
            except Exception as e:
                print(f"[!] Compilation error: {e}")

        self.is_sdxl = hasattr(self.pipe, "text_encoder_2")
        self._warmup()
        print(f"[NeuroRender] {self.mode.upper()} Ready. (Architecture: {'SDXL' if self.is_sdxl else 'Standard SD'})")

    def _warmup(self):
        dummy_image = Image.fromarray(np.zeros((GH, GW, 3), dtype=np.uint8))
        with torch.no_grad():
            for _ in range(2):
                if self.is_sdxl:
                    p_emb, _, pool, _ = self.pipe.encode_prompt(
                        prompt="warmup", device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False
                    )
                    self.generate(image=dummy_image, prompt_embeds=p_emb, pooled_prompt_embeds=pool, strength=0.7)
                else:
                    p_emb, _ = self.pipe.encode_prompt(
                        prompt="warmup", device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False
                    )
                    self.generate(image=dummy_image, prompt_embeds=p_emb, strength=0.7)

    def encode_prompts_universal(self, prompt_list):
        p_embeds_list = []
        pooled_list = []
        with torch.no_grad():
            for p in prompt_list:
                if self.is_sdxl:
                    p_emb, _, pool, _ = self.pipe.encode_prompt(
                        prompt=p, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False
                    )
                    p_embeds_list.append(p_emb[0].cpu().numpy())
                    pooled_list.append(pool[0].cpu().numpy())
                else:
                    p_emb, _ = self.pipe.encode_prompt(
                        prompt=p, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False
                    )
                    p_embeds_list.append(p_emb[0].cpu().numpy())
                    
        return {
            'c_bases': p_embeds_list,
            'pooled_bases': pooled_list if self.is_sdxl else None,
            'is_sdxl': self.is_sdxl,
            'mode': self.mode
        }

    def generate(self, **kwargs):
        if 'strength' in kwargs:
            kwargs['strength'] = float(np.clip(kwargs['strength'], 0.35, 0.95))

        if self.mode in ["turbo", "sd-turbo", "sdxl-turbo", "sdxl"]:
            kwargs.setdefault("strength", 0.5)
            kwargs.setdefault("num_inference_steps", 2)
            kwargs.setdefault("guidance_scale", 0.0)
        else:
            kwargs.setdefault("strength", 0.5)
            kwargs.setdefault("num_inference_steps", 4)
            kwargs.setdefault("guidance_scale", 1.0)
            
        return self.pipe(**kwargs).images[0]

