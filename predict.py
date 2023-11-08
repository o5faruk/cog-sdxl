import hashlib
import json
import os
import shutil
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from weights import WeightsDownloadCache

import numpy as np
import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    # AutoencoderKL,
)
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.utils import load_image
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import CLIPImageProcessor

from dataset_and_utils import TokenEmbeddingsHandler
from preprocess import (
    clipseg_mask_generator,
    crop_faces_to_square,
    paste_inpaint_into_original_image,
    face_mask_google_mediapipe,
)
from download_weights import download_weights

from PIL import Image


SDXL_MODEL_CACHE = "./sdxl-cache"
# REFINER_MODEL_CACHE = "./refiner-cache"
SAFETY_CACHE = "./safety-cache"
FEATURE_EXTRACTOR = "./feature-extractor"
SDXL_URL = "https://weights.replicate.delivery/default/sdxl/sdxl-vae-upcast-fix.tar"
# REFINER_URL = (
#     "https://weights.replicate.delivery/default/sdxl/refiner-no-vae-no-encoder-1.0.tar"
# )
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"


class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)


SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KarrasDPM": KarrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}


class Predictor(BasePredictor):
    def load_trained_weights(self, weights, pipe):
        from no_init import no_init_or_tensor

        # weights can be a URLPath, which behaves in unexpected ways
        weights = str(weights)
        if self.tuned_weights == weights:
            print("skipping loading .. weights already loaded")
            return

        self.tuned_weights = weights

        local_weights_cache = self.weights_cache.ensure(weights)

        # load UNET
        print("Loading fine-tuned model")
        self.is_lora = False

        maybe_unet_path = os.path.join(local_weights_cache, "unet.safetensors")
        if not os.path.exists(maybe_unet_path):
            print("Does not have Unet. assume we are using LoRA")
            self.is_lora = True

        if not self.is_lora:
            print("Loading Unet")

            new_unet_params = load_file(
                os.path.join(local_weights_cache, "unet.safetensors")
            )
            # this should return _IncompatibleKeys(missing_keys=[...], unexpected_keys=[])
            pipe.unet.load_state_dict(new_unet_params, strict=False)

        else:
            print("Loading Unet LoRA")

            unet = pipe.unet

            tensors = load_file(os.path.join(local_weights_cache, "lora.safetensors"))

            unet_lora_attn_procs = {}
            name_rank_map = {}
            for tk, tv in tensors.items():
                # up is N, d
                if tk.endswith("up.weight"):
                    proc_name = ".".join(tk.split(".")[:-3])
                    r = tv.shape[1]
                    name_rank_map[proc_name] = r

            for name, attn_processor in unet.attn_processors.items():
                cross_attention_dim = (
                    None
                    if name.endswith("attn1.processor")
                    else unet.config.cross_attention_dim
                )
                if name.startswith("mid_block"):
                    hidden_size = unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(unet.config.block_out_channels))[
                        block_id
                    ]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = unet.config.block_out_channels[block_id]
                with no_init_or_tensor():
                    module = LoRAAttnProcessor2_0(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        rank=name_rank_map[name],
                    )
                unet_lora_attn_procs[name] = module.to("cuda", non_blocking=True)

            unet.set_attn_processor(unet_lora_attn_procs)
            unet.load_state_dict(tensors, strict=False)

        # load text
        handler = TokenEmbeddingsHandler(
            [pipe.text_encoder, pipe.text_encoder_2], [pipe.tokenizer, pipe.tokenizer_2]
        )
        handler.load_embeddings(os.path.join(local_weights_cache, "embeddings.pti"))

        # load params
        with open(os.path.join(local_weights_cache, "special_params.json"), "r") as f:
            params = json.load(f)
        self.token_map = params

        self.tuned_model = True

    def setup(self, weights: Optional[Path] = None):
        """Load the model into memory to make running multiple predictions efficient"""

        start = time.time()
        self.tuned_model = False
        self.tuned_weights = None
        if str(weights) == "weights":
            weights = None

        self.weights_cache = WeightsDownloadCache()

        print("Loading safety checker...")
        if not os.path.exists(SAFETY_CACHE):
            download_weights(SAFETY_URL, SAFETY_CACHE)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_CACHE, torch_dtype=torch.float16
        ).to("cuda")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)

        if not os.path.exists(SDXL_MODEL_CACHE):
            download_weights(SDXL_URL, SDXL_MODEL_CACHE)

        print("Loading sdxl txt2img pipeline...")
        self.txt2img_pipe = DiffusionPipeline.from_pretrained(
            SDXL_MODEL_CACHE,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.is_lora = False
        if weights or os.path.exists("./trained-model"):
            self.load_trained_weights(weights, self.txt2img_pipe)

        self.txt2img_pipe.to("cuda")

        print("Loading SDXL img2img pipeline...")
        self.img2img_pipe = StableDiffusionXLImg2ImgPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            text_encoder_2=self.txt2img_pipe.text_encoder_2,
            tokenizer=self.txt2img_pipe.tokenizer,
            tokenizer_2=self.txt2img_pipe.tokenizer_2,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
        )
        self.img2img_pipe.to("cuda")

        print("Loading SDXL inpaint pipeline...")
        self.inpaint_pipe = StableDiffusionXLInpaintPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            text_encoder_2=self.txt2img_pipe.text_encoder_2,
            tokenizer=self.txt2img_pipe.tokenizer,
            tokenizer_2=self.txt2img_pipe.tokenizer_2,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
        )
        self.inpaint_pipe.to("cuda")

        # print("Loading SDXL refiner pipeline...")
        # FIXME(ja): should the vae/text_encoder_2 be loaded from SDXL always?
        #            - in the case of fine-tuned SDXL should we still?
        # FIXME(ja): if the answer to above is use VAE/Text_Encoder_2 from fine-tune
        #            what does this imply about lora + refiner? does the refiner need to know about

        # if not os.path.exists(REFINER_MODEL_CACHE):
        #     download_weights(REFINER_URL, REFINER_MODEL_CACHE)

        # print("Loading refiner pipeline...")
        # self.refiner = DiffusionPipeline.from_pretrained(
        #     REFINER_MODEL_CACHE,
        #     text_encoder_2=self.txt2img_pipe.text_encoder_2,
        #     vae=self.txt2img_pipe.vae,
        #     torch_dtype=torch.float16,
        #     use_safetensors=True,
        #     variant="fp16",
        # )
        # self.refiner.to("cuda")

        print("Loading controlnet model")
        self.controlnet = ControlNetModel.from_pretrained(
            "thibaud/controlnet-openpose-sdxl-1.0",
            torch_dtype=torch.float16,
            cache_dir="/src/controlnet-cache",
        )
        self.controlnet.to("cuda")

        print("Loading XL Controlnet pipe")
        self.controlnet_pipe = StableDiffusionXLControlNetPipeline(
            controlnet=self.controlnet,
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            text_encoder_2=self.txt2img_pipe.text_encoder_2,
            tokenizer=self.txt2img_pipe.tokenizer,
            tokenizer_2=self.txt2img_pipe.tokenizer_2,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
        )
        self.controlnet_pipe.to("cuda")

        print("setup took: ", time.time() - start)
        # self.txt2img_pipe.__class__.encode_prompt = new_encode_prompt

    def load_image(self, path):
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")

    def run_safety_checker(self, image):
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(
            "cuda"
        )
        np_image = [np.array(val) for val in image]
        image, has_nsfw_concept = self.safety_checker(
            images=np_image,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return image, has_nsfw_concept

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="a TOK man in preppy style, old money aesthetic, posh style, elite school stlye, luxurious style, gossip girl neo-prep style, ralph lauren style, country club style, ivy league style",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="plastic, blurry, grainy, [deformed | disfigured], poorly drawn, [bad : wrong] anatomy, [extra | missing | floating | disconnected] limb, (mutated hands and fingers), blurry",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=25
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=9
        ),
        inpaint_prompt: str = Input(
            description="Input inpaint prompt",
            default="closeup of TOK man, natural skin, skin moles, 37 y o caucasian",
        ),
        inpaint_negative_prompt: str = Input(
            description="Input inpaint negative prompt",
            default="plastic, blurry, grainy, [deformed | disfigured], poorly drawn, [bad : wrong] anatomy, [extra | missing | floating | disconnected] limb, (mutated hands and fingers), blurry",
        ),
        inpaint_num_inference_steps: int = Input(
            description="Number of denoising steps for inpainting",
            ge=1,
            le=500,
            default=25,
        ),
        second_inpaint_num_inference_steps: int = Input(
            description="Number of denoising steps for inpainting",
            ge=1,
            le=500,
            default=25,
        ),
        inpaint_guidance_scale: float = Input(
            description="Scale for classifier-free guidance for inpainting",
            ge=1,
            le=50,
            default=9,
        ),
        second_inpaint_guidance_scale: float = Input(
            description="Scale for classifier-free guidance for inpainting",
            ge=1,
            le=50,
            default=9,
        ),
        inpaint_strength: float = Input(
            description="Prompt strength when using inpaint. 1.0 corresponds to full destruction of information in image",
            ge=0.0,
            le=1.0,
            default=0.35,
        ),
        second_inpaint_strength: float = Input(
            description="Prompt strength when using inpaint. 1.0 corresponds to full destruction of information in image",
            ge=0.0,
            le=1.0,
            default=0.35,
        ),
        upscale_by: float = Input(
            description="Upscale by factor",
            ge=1,
            le=4,
            default=1.5,
        ),
        upscale_prompt: str = Input(
            description="Input inpaint prompt",
            default="closeup of TOK man, natural skin, skin moles, 37 y o caucasian",
        ),
        upscale_negative_prompt: str = Input(
            description="Input inpaint negative prompt",
            default="plastic, blurry, grainy, [deformed | disfigured], poorly drawn, [bad : wrong] anatomy, [extra | missing | floating | disconnected] limb, (mutated hands and fingers), blurry",
        ),
        upscale_num_inference_steps: int = Input(
            description="Number of denoising steps for inpainting",
            ge=1,
            le=500,
            default=25,
        ),
        upscale_guidance_scale: float = Input(
            description="Scale for classifier-free guidance for inpainting",
            ge=1,
            le=50,
            default=9,
        ),
        upscale_strength: float = Input(
            description="Prompt strength when using inpaint. 1.0 corresponds to full destruction of information in image",
            ge=0.0,
            le=1.0,
            default=0.35,
        ),
        width: int = Input(
            description="Width of output image",
            default=768,
        ),
        height: int = Input(
            description="Height of output image",
            default=1024,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=SCHEDULERS.keys(),
            default="K_EULER",
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using img2img / inpaint. 1.0 corresponds to full destruction of information in image",
            ge=0.0,
            le=1.0,
            default=0.8,
        ),
        mask_blur_amount: float = Input(
            description="Amount to blur the inpaint mask by",
            default=8.0,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=1234
        ),
        lora_scale: float = Input(
            description="LoRA additive scale. Only applicable on trained models.",
            ge=0.0,
            le=1.0,
            default=0.4,
        ),
        pose_image: Path = Input(
            description="pose_image",
            default=None,
        ),
        controlnet_conditioning_scale: float = Input(
            description="controlnet_conditioning_scale",
            ge=0.0,
            le=1.0,
            default=0.6,
        ),
        weights: str = Input(
            description="Replicate LoRA weights to use. Leave blank to use the default weights.",
            default="https://replicate.delivery/pbxt/cv7B5U7T4G4kFN8eXIg7F8FRfbC54hIWyeizfNuH9lcBevtOC/trained_model.tar",
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if weights:
            self.load_trained_weights(weights, self.txt2img_pipe)
            self.load_trained_weights(weights, self.inpaint_pipe)

        # OOMs can leave vae in bad state
        if self.txt2img_pipe.vae.dtype == torch.float32:
            self.txt2img_pipe.vae.to(dtype=torch.float16)

        sdxl_kwargs = {}
        if self.tuned_model:
            # consistency with fine-tuning API
            for k, v in self.token_map.items():
                prompt = prompt.replace(k, v)
        print(f"Prompt: {prompt}")
        if pose_image:
            print("pose mode")
            sdxl_kwargs["image"] = self.load_image(pose_image)
            sdxl_kwargs["controlnet_conditioning_scale"] = controlnet_conditioning_scale
            sdxl_kwargs["width"] = width
            sdxl_kwargs["height"] = height
            pipe = self.controlnet_pipe
        else:
            print("txt2img mode")
            sdxl_kwargs["width"] = width
            sdxl_kwargs["height"] = height
            pipe = self.txt2img_pipe

        pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config)
        generator = torch.Generator("cuda").manual_seed(seed)

        common_args = {
            "prompt": [prompt] * num_outputs,
            "negative_prompt": [negative_prompt] * num_outputs,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }

        if self.is_lora:
            sdxl_kwargs["cross_attention_kwargs"] = {"scale": lora_scale}

        # Print kwargs
        print("common_args: ", common_args)
        print("sdxl_kwargs: ", sdxl_kwargs)

        torch.cuda.empty_cache()
        with torch.no_grad():
            output = pipe(**common_args, **sdxl_kwargs)

        output_paths = []

        google_face_masks = face_mask_google_mediapipe(
            images=output.images,
            blur_amount=mask_blur_amount
        )

        # Add google_face_masks to output_paths
        for i, mask in enumerate(google_face_masks):
            output_path = f"/tmp/google-mask-{i}.png"
            mask.save(output_path)
            output_paths.append(Path(output_path))

        self.inpaint_pipe.scheduler = SCHEDULERS[scheduler].from_config(
            pipe.scheduler.config
        )

        # Add output images to output_paths
        for i, image in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        cropped_face, cropped_mask, left_top, orig_size = crop_faces_to_square(
            output.images[0], google_face_masks[0]
        )

        # Add cropped face to output_paths
        cropped_face_path = f"/tmp/cropped_face.png"
        cropped_face.save(cropped_face_path)
        output_paths.append(Path(cropped_face_path))

        # Add cropped mask to output_paths
        cropped_mask_path = f"/tmp/cropped_mask.png"
        cropped_mask.save(cropped_mask_path)
        output_paths.append(Path(cropped_mask_path))

        print("inpainting mode")

        self.inpaint_pipe.scheduler = SCHEDULERS[scheduler].from_config(
            self.inpaint_pipe.scheduler.config
        )

        # # If inpaint_prompt has length, do inpainting
        # if len(inpaint_prompt) > 0:
        # Print combined args
        inpaint_kwargs = {
            "prompt": inpaint_prompt,
            "negative_prompt": inpaint_negative_prompt,
            "guidance_scale": inpaint_guidance_scale,
            "generator": torch.Generator("cuda").manual_seed(seed),
            "num_inference_steps": inpaint_num_inference_steps,
            "width": cropped_face.width,
            "height": cropped_face.height,
            "output_type": "pil",
            "image": cropped_face,
            "mask_image": cropped_mask,
            "strength": inpaint_strength,
        }

        output_image = output.images[0]

        inpaint_output = self.inpaint_pipe(**inpaint_kwargs)

        # Add inpaint outputs to output_paths
        inpaint_output_path = f"/tmp/inpaint-out-{0}.png"
        inpaint_output.images[0].save(inpaint_output_path)
        output_paths.append(Path(inpaint_output_path))

        inpaint_image = inpaint_output.images[0]

        # Inpaint again

        inpaint_kwargs = {
            "prompt": inpaint_prompt,
            "negative_prompt": inpaint_negative_prompt,
            "guidance_scale": second_inpaint_guidance_scale,
            "generator": torch.Generator("cuda").manual_seed(seed),
            "num_inference_steps": second_inpaint_num_inference_steps,
            "width": inpaint_image.width,
            "height": inpaint_image.height,
            "output_type": "pil",
            "image": inpaint_image,
            "mask_image": cropped_mask,
            "strength": second_inpaint_strength,
        }

        second_inpaint_output = self.inpaint_pipe(**inpaint_kwargs)

        # Add inpaint outputs to output_paths
        second_inpaint_output_path = f"/tmp/second-inpaint-out-{0}.png"
        second_inpaint_output.images[0].save(second_inpaint_output_path)
        output_paths.append(Path(second_inpaint_output_path))

        inpaint_image = second_inpaint_output.images[0]

        pasted_image = paste_inpaint_into_original_image(
            output_image,
            cropped_mask,
            left_top,
            inpaint_image,
            orig_size,
        )

        output_path = f"/tmp/final-out-{i}.png"
        pasted_image.save(output_path)
        output_paths.append(Path(output_path))

        # print pasted_image size
        # print("pasted_image size: ", pasted_image.size)

        # Do img2img pass
        # self.img2img_pipe.scheduler = SCHEDULERS[scheduler].from_config(
        #     self.img2img_pipe.scheduler.config
        # )

        # img2img_kwargs = {
        #     "prompt": upscale_prompt,
        #     "negative_prompt": upscale_negative_prompt,
        #     "guidance_scale": upscale_guidance_scale,
        #     "generator": torch.Generator("cuda").manual_seed(seed),
        #     "num_inference_steps": upscale_num_inference_steps,
        #     "output_type": "pil",
        #     "image": pasted_image,
        #     "strength": upscale_strength,
        # }

        # img2img_output = self.img2img_pipe(**img2img_kwargs)

        # img2img_result = img2img_output.images[0]

        # Add img2img output to output_paths
        # img2img_output_path = f"/tmp/img2img-out-{0}.png"
        # img2img_result.save(img2img_output_path)
        # output_paths.append(Path(img2img_output_path))

        return output_paths
