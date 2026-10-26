import gc
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import CLIPTextModel
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    AutoPipelineForImage2Image,
    LCMScheduler,
    AutoencoderTiny
)
from neuro_amodal_engine import HardwareVRAMBus

GW, GH = 512, 384


class NeuroRender:
    def __init__(self, mode="lcm", compile_unet=False, remote_conn=None, use_taesd=True, deterministic=False, seed=0):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16
        self.mode = mode.lower().replace("_", "-")
        self.remote_conn = remote_conn
        self.use_taesd = use_taesd
        self.deterministic = bool(deterministic)
        self.base_seed = int(seed)
        self.step_counter = 0
        self.vae_full = None
        self.vae_tiny = None
        self.vram_bus = HardwareVRAMBus()

        self.proj_768_to_1024 = None
        self.proj_768_to_1280 = None
        self.proj_768_to_pooled = None

        if self.remote_conn is not None:
            return

        print(f"⏳ [NeuroRender] Загрузка модели [{self.mode.upper()}] на {self.device} (Stochastic={not self.deterministic}, Seed={self.base_seed})...")

        if self.mode in ["sdxl-turbo", "sdxl"]:
            self.mode = "sdxl-turbo"
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                "stabilityai/sdxl-turbo",
                torch_dtype=self.dtype,
                variant="fp16"
            ).to(self.device)
            self._build_sdxl_bridge()

        elif self.mode in ["turbo", "sd-turbo"]:
            self.mode = "sd-turbo"
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                "stabilityai/sd-turbo",
                torch_dtype=self.dtype,
                variant="fp16"
            ).to(self.device)
            self._build_sdturbo_bridge()

        else:
            self.mode = "lcm"
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "SimianLuo/LCM_Dreamshaper_v7",
                torch_dtype=self.dtype
            ).to(self.device)
            self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

        self.is_sdxl = hasattr(self.pipe, "text_encoder_2")

        self.vae_full = self.pipe.vae
        try:
            taesd_id = "madebyollin/taesdxl" if self.is_sdxl else "madebyollin/taesd"
            self.vae_tiny = AutoencoderTiny.from_pretrained(taesd_id, torch_dtype=self.dtype).to(self.device)
        except Exception as e:
            print(f"[!] TAESD load note: {e}")
            self.vae_tiny = None

        self.set_taesd(self.use_taesd)

        self.pipe.safety_checker = None
        self.pipe.set_progress_bar_config(disable=True)

        if compile_unet and self.device == 'cuda':
            try:
                self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=False)
            except Exception:
                pass

        self._warmup()
        print(f"✅ [NeuroRender] Модель {self.mode.upper()} готова! (Intra-Step VRAM Live Thought Injection включен)")

    @torch.no_grad()
    def _build_sdxl_bridge(self):
        try:
            w_l = self.pipe.text_encoder.get_input_embeddings().weight.detach().float()
            w_g = self.pipe.text_encoder_2.get_input_embeddings().weight.detach().float()
            n = min(w_l.shape[0], w_g.shape[0])
            sol = torch.linalg.lstsq(w_l[:n], w_g[:n]).solution
            self.proj_768_to_1280 = sol.to(device=self.device, dtype=self.dtype)
            if hasattr(self.pipe.text_encoder_2, "text_projection"):
                t_proj = self.pipe.text_encoder_2.text_projection.weight.detach().float()
                self.proj_768_to_pooled = (sol @ t_proj.T).to(device=self.device, dtype=self.dtype)
            else:
                self.proj_768_to_pooled = self.proj_768_to_1280
        except Exception as e:
            print(f"⚠️ [SDXL Bridge Warning]: {e}")

    @torch.no_grad()
    def _build_sdturbo_bridge(self):
        try:
            clip_l = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float32).to(self.device)
            w_l = clip_l.get_input_embeddings().weight.detach().float()
            w_h = self.pipe.text_encoder.get_input_embeddings().weight.detach().float()
            n = min(w_l.shape[0], w_h.shape[0])
            sol = torch.linalg.lstsq(w_l[:n], w_h[:n]).solution
            self.proj_768_to_1024 = sol.to(device=self.device, dtype=self.dtype)
            del clip_l
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"⚠️ [SD-Turbo Bridge Warning]: {e}")

    def set_taesd(self, enable_taesd: bool):
        self.use_taesd = bool(enable_taesd)
        if self.use_taesd and self.vae_tiny is not None:
            self.pipe.vae = self.vae_tiny
        elif self.vae_full is not None:
            self.pipe.vae = self.vae_full

    @torch.no_grad()
    def adapt_prompt_embeds(self, t_emb: torch.Tensor, pooled_emb: torch.Tensor | None = None):
        if t_emb.ndim == 2:
            t_emb = t_emb.unsqueeze(0)
        t_emb = t_emb.to(device=self.device, dtype=self.dtype)
        dim = t_emb.shape[-1]

        if self.is_sdxl:
            if dim == 768:
                if self.proj_768_to_1280 is not None:
                    emb_1280 = torch.matmul(t_emb, self.proj_768_to_1280)
                    norm_ref = torch.norm(t_emb, p=2, dim=-1, keepdim=True) * (1280.0 / 768.0) ** 0.5
                    emb_1280 = F.normalize(emb_1280, p=2, dim=-1) * norm_ref
                else:
                    emb_1280 = F.pad(t_emb, (0, 1280 - 768))
                if pooled_emb is None:
                    if self.proj_768_to_pooled is not None:
                        pooled_emb = torch.matmul(t_emb[:, -1, :], self.proj_768_to_pooled)
                    else:
                        pooled_emb = emb_1280.mean(dim=1)
                t_emb = torch.cat([t_emb, emb_1280], dim=-1)
            elif pooled_emb is None:
                pooled_emb = torch.zeros((t_emb.shape[0], 1280), dtype=self.dtype, device=self.device)
            return t_emb, pooled_emb

        elif self.mode == "sd-turbo":
            if dim == 768:
                if self.proj_768_to_1024 is not None:
                    emb_1024 = torch.matmul(t_emb, self.proj_768_to_1024)
                    norm_ref = torch.norm(t_emb, p=2, dim=-1, keepdim=True) * (1024.0 / 768.0) ** 0.5
                    t_emb = F.normalize(emb_1024, p=2, dim=-1) * norm_ref
                else:
                    t_emb = F.pad(t_emb, (0, 1024 - 768))
            return t_emb, None

        return t_emb, None

    def _warmup(self):
        dummy_image = Image.fromarray(np.zeros((GH, GW, 3), dtype=np.uint8))
        dummy_768 = torch.randn((1, 77, 768), device=self.device, dtype=self.dtype)
        with torch.no_grad():
            self.generate(image=dummy_image, prompt_embeds=dummy_768, strength=0.7, num_inference_steps=2)

    def encode_prompts_universal(self, prompt_list):
        p_embeds_list, pooled_list = [], []
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
        req_det = kwargs.pop('deterministic', None)
        req_stoch = kwargs.pop('stochastic', None)
        req_seed = kwargs.pop('seed', None)
        wm_tensor = kwargs.pop('wm_5x5x32x120', None)

        # Подхватываем общую VRAM-шину без сокетов и копирований через CPU!
        self.vram_bus.try_attach_to_existing_vram()
        has_vram_bus = (self.vram_bus.attached_pid > 0)

        if req_stoch is not None:
            use_det = not bool(req_stoch)
        elif req_det is not None:
            use_det = bool(req_det)
        else:
            use_det = self.deterministic

        active_seed = int(req_seed) if req_seed is not None else self.base_seed

        if 'use_taesd' in kwargs:
            req_t = bool(kwargs.pop('use_taesd'))
            if req_t != self.use_taesd:
                self.set_taesd(req_t)

        # Если подключена Zero-Copy VRAM шина и prompt_embeds не передан явно с CPU, берем прямо из VRAM!
        if ('prompt_embeds' not in kwargs or kwargs['prompt_embeds'] is None) and has_vram_bus:
            kwargs['prompt_embeds'] = self.vram_bus.live_clip_77x768.unsqueeze(0)

        if wm_tensor is None and has_vram_bus:
            wm_tensor = self.vram_bus.wm_field_5x5x32x120

        if 'prompt_embeds' in kwargs and kwargs['prompt_embeds'] is not None:
            t_emb, pool_emb = self.adapt_prompt_embeds(
                kwargs['prompt_embeds'],
                kwargs.get('pooled_prompt_embeds')
            )
            kwargs['prompt_embeds'] = t_emb
            if self.is_sdxl:
                kwargs['pooled_prompt_embeds'] = pool_emb
            else:
                kwargs.pop('pooled_prompt_embeds', None)

        steps = int(kwargs.get("num_inference_steps", 2))
        strength = float(np.clip(kwargs.get("strength", 0.6), 0.25, 0.98))
        if int(steps * strength) < 1:
            steps = max(2, int(np.ceil(1.0 / max(0.25, strength))))
        kwargs["num_inference_steps"] = steps
        kwargs["strength"] = strength

        if self.mode in ["turbo", "sd-turbo", "sdxl-turbo", "sdxl"]:
            kwargs["guidance_scale"] = 0.0
        else:
            kwargs.setdefault("guidance_scale", 1.35)

        self.step_counter += 1

        if use_det:
            pe = kwargs.get("prompt_embeds")
            if wm_tensor is not None:
                weights = torch.linspace(0.5, 2.5, wm_tensor.numel(), device=wm_tensor.device, dtype=torch.float32).view_as(wm_tensor)
                sig = float(torch.sum(wm_tensor * weights).item()) * 100000.0
                wm_seed = (int(abs(round(sig))) + active_seed) % 2147483640 + 1
            elif pe is not None:
                weights = torch.linspace(0.5, 2.5, pe.numel(), device=pe.device, dtype=torch.float32).view_as(pe)
                sig = float(torch.sum(pe.float() * weights).item()) * 10000.0
                wm_seed = (int(abs(round(sig))) + active_seed) % 2147483640 + 1
            else:
                wm_seed = 42
            kwargs["generator"] = torch.Generator(device=self.device).manual_seed(wm_seed)
        else:
            if active_seed > 0:
                kwargs["generator"] = torch.Generator(device=self.device).manual_seed(active_seed + self.step_counter)
            else:
                kwargs.pop("generator", None)

        # ВНУТРИШАГОВАЯ ПОДСТАНОВКА МЫСЛИ (INTRA-STEP LIVE THOUGHT INJECTION):
        # Между шагами денойзинга U-Net (например, между шагом 1 и шагом 2/3) мы прямо в VRAM
        # обновляем prompt_embeds из живой рабочей памяти live_clip_77x768, не дожидаясь конца кадра!
        if has_vram_bus and steps > 1:
            def _live_wm_step_callback(pipe_obj, step_index, timestep, callback_kwargs):
                live_77 = self.vram_bus.live_clip_77x768
                if torch.norm(live_77) > 1e-3:
                    new_pe, _ = self.adapt_prompt_embeds(live_77.unsqueeze(0))
                    if "prompt_embeds" in callback_kwargs and callback_kwargs["prompt_embeds"] is not None:
                        old_pe = callback_kwargs["prompt_embeds"]
                        if old_pe.shape == new_pe.shape:
                            callback_kwargs["prompt_embeds"] = old_pe * 0.40 + new_pe * 0.60
                return callback_kwargs

            kwargs["callback_on_step_end"] = _live_wm_step_callback
            kwargs["callback_on_step_end_tensor_inputs"] = ["latents", "prompt_embeds"]

        return self.pipe(**kwargs).images[0]
