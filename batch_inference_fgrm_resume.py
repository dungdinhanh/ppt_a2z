import argparse
import os
import random
import tqdm
from torch.utils.data.distributed import DistributedSampler
import cv2
import gradio as gr
import numpy as np
import torch
from controlnet_aux import HEDdetector, OpenposeDetector
from PIL import Image, ImageFilter
from safetensors.torch import load_model
from transformers import CLIPTextModel, DPTFeatureExtractor, DPTForDepthEstimation
from torchvision.utils import save_image
from diffusers import UniPCMultistepScheduler
from diffusers.pipelines.controlnet.pipeline_controlnet import ControlNetModel
from powerpaint.models.BrushNet_CA import BrushNetModel
from powerpaint.models.unet_2d_condition import UNet2DConditionModel
from powerpaint.pipelines.pipeline_PowerPaint import StableDiffusionInpaintPipeline as Pipeline
from powerpaint.pipelines.pipeline_PowerPaint_Brushnet_CA import StableDiffusionPowerPaintBrushNetPipeline
from powerpaint.pipelines.pipeline_PowerPaint_ControlNet import (
    StableDiffusionControlNetInpaintPipeline as controlnetPipeline,
)
from powerpaint.utils.utils import TokenizerWrapper, add_tokens
from torch.utils.data import Dataset
import os
from PIL import Image
# import albumentations as 
from accelerate import Accelerator
import pandas as pd
torch.set_grad_enabled(False)
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def add_task(prompt, negative_prompt, control_type, version):
    pos_prefix = neg_prefix = ""
    if control_type == "object-removal" or control_type == "image-outpainting":
        if version == "ppt-v1":
            pos_prefix = "empty scene blur " + prompt
            neg_prefix = negative_prompt
        promptA = pos_prefix + " P_ctxt"
        promptB = pos_prefix + " P_ctxt"
        negative_promptA = neg_prefix + " P_obj"
        negative_promptB = neg_prefix + " P_obj"
    elif control_type == "shape-guided":
        if version == "ppt-v1":
            pos_prefix = prompt
            neg_prefix = negative_prompt + ", worst quality, low quality, normal quality, bad quality, blurry "
        promptA = pos_prefix + " P_shape"
        promptB = pos_prefix + " P_ctxt"
        negative_promptA = neg_prefix + "P_shape"
        negative_promptB = neg_prefix + "P_ctxt"
    else:
        if version == "ppt-v1":
            pos_prefix = prompt
            neg_prefix = negative_prompt + ", worst quality, low quality, normal quality, bad quality, blurry "
        promptA = pos_prefix + " P_obj"
        promptB = pos_prefix + " P_obj"
        negative_promptA = neg_prefix + "P_obj"
        negative_promptB = neg_prefix + "P_obj"

    return promptA, promptB, negative_promptA, negative_promptB


def select_tab_text_guided():
    return "text-guided"


def select_tab_object_removal():
    return "object-removal"


def select_tab_image_outpainting():
    return "image-outpainting"


def select_tab_shape_guided():
    return "shape-guided"


class PowerPaintController:
    def __init__(self, weight_dtype, checkpoint_dir, local_files_only, version, device) -> None:
        self.version = version
        self.checkpoint_dir = checkpoint_dir
        self.local_files_only = local_files_only

        # initialize powerpaint pipeline
        if version == "ppt-v1":
            self.pipe = Pipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting", torch_dtype=weight_dtype, local_files_only=local_files_only
            )
            self.pipe.tokenizer = TokenizerWrapper(
                from_pretrained="runwayml/stable-diffusion-v1-5",
                subfolder="tokenizer",
                revision=None,
                local_files_only=local_files_only,
            )

            # add learned task tokens into the tokenizer
            add_tokens(
                tokenizer=self.pipe.tokenizer,
                text_encoder=self.pipe.text_encoder,
                placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
                initialize_tokens=["a", "a", "a"],
                num_vectors_per_token=10,
            )

            # loading pre-trained weights
            load_model(self.pipe.unet, os.path.join(checkpoint_dir, "unet/unet.safetensors"))
            load_model(self.pipe.text_encoder, os.path.join(checkpoint_dir, "text_encoder/text_encoder.safetensors"))
            self.pipe = self.pipe.to("cuda")

            # initialize controlnet-related models
            self.depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
            self.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
            self.openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            self.hed = HEDdetector.from_pretrained("lllyasviel/ControlNet")

            base_control = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny", torch_dtype=weight_dtype, local_files_only=local_files_only
            )
            self.control_pipe = controlnetPipeline(
                self.pipe.vae,
                self.pipe.text_encoder,
                self.pipe.tokenizer,
                self.pipe.unet,
                base_control,
                self.pipe.scheduler,
                None,
                None,
                False,
            )
            # self.control_pipe = self.control_pipe.to("cuda")

            self.current_control = "canny"
            # controlnet_conditioning_scale = 0.8
        else:
            # brushnet-based version
            unet = UNet2DConditionModel.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="unet",
                revision=None,
                torch_dtype=weight_dtype,
                local_files_only=local_files_only,
            )
            text_encoder_brushnet = CLIPTextModel.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="text_encoder",
                revision=None,
                torch_dtype=weight_dtype,
                local_files_only=local_files_only,
            )
            brushnet = BrushNetModel.from_unet(unet)
            base_model_path = os.path.join(checkpoint_dir, "realisticVisionV60B1_v51VAE")
            self.pipe = StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
                base_model_path,
                brushnet=brushnet,
                text_encoder_brushnet=text_encoder_brushnet,
                torch_dtype=weight_dtype,
                low_cpu_mem_usage=False,
                safety_checker=None,
            )
            self.pipe.unet = UNet2DConditionModel.from_pretrained(
                base_model_path,
                subfolder="unet",
                revision=None,
                torch_dtype=weight_dtype,
                local_files_only=local_files_only,
            )
            self.pipe.tokenizer = TokenizerWrapper(
                from_pretrained=base_model_path,
                subfolder="tokenizer",
                revision=None,
                torch_type=weight_dtype,
                local_files_only=local_files_only,
            )

            # add learned task tokens into the tokenizer
            add_tokens(
                tokenizer=self.pipe.tokenizer,
                text_encoder=self.pipe.text_encoder_brushnet,
                placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
                initialize_tokens=["a", "a", "a"],
                num_vectors_per_token=10,
            )
            load_model(
                self.pipe.brushnet,
                os.path.join(checkpoint_dir, "PowerPaint_Brushnet/diffusion_pytorch_model.safetensors"),
            )
            
            state_dict = torch.load(os.path.join(checkpoint_dir, "PowerPaint_Brushnet/pytorch_model.bin"), map_location=device)
            self.pipe.text_encoder_brushnet.load_state_dict(state_dict, strict=False)

            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

            # self.pipe.enable_model_cpu_offload()
            # self.pipe = self.pipe.to("cuda")
    
    def set_eval_mode(self):
        if self.version == "ppt-v1":
            pipe = self.pipe
            if hasattr(pipe, "unet"): pipe.unet.eval()
            if hasattr(pipe, "text_encoder"): pipe.text_encoder.eval()
            if hasattr(pipe, "vae"): pipe.vae.eval()

            if hasattr(self.controller, "control_pipe"):
                cp = self.controller.control_pipe
                if hasattr(cp, "unet"): cp.unet.eval()
                if hasattr(cp, "vae"): cp.vae.eval()
                if hasattr(cp, "text_encoder"): cp.text_encoder.eval()
                if hasattr(cp, "controlnet"): cp.controlnet.eval()

        else:
            pipe = self.pipe
            if hasattr(pipe, "unet"): pipe.unet.eval()
            if hasattr(pipe, "vae"): pipe.vae.eval()
            if hasattr(pipe, "text_encoder_brushnet"): pipe.text_encoder_brushnet.eval()
            if hasattr(pipe, "brushnet"): pipe.brushnet.eval()


    def get_depth_map(self, image):
        # image = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
        with torch.no_grad(), torch.autocast("cuda"):
            depth_map = self.depth_estimator(image).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)

        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        return image

    def load_controlnet(self, control_type):
        if self.current_control != control_type:
            if control_type == "canny" or control_type is None:
                self.control_pipe.controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-canny", torch_dtype=weight_dtype, local_files_only=self.local_files_only
                )
            elif control_type == "pose":
                self.control_pipe.controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-openpose",
                    torch_dtype=weight_dtype,
                    local_files_only=self.local_files_only,
                )
            elif control_type == "depth":
                self.control_pipe.controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-depth", torch_dtype=weight_dtype, local_files_only=self.local_files_only
                )
            else:
                self.control_pipe.controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-hed", torch_dtype=weight_dtype, local_files_only=self.local_files_only
                )
            # self.control_pipe = self.control_pipe.to("cuda")
            self.current_control = control_type

    def predict(
        self,
        input_image,
        prompt,
        fitting_degree,
        ddim_steps,
        scale,
        seed,
        negative_prompt,
        task,
        vertical_expansion_ratio,
        horizontal_expansion_ratio,
        device
    ):
        size1, size2 = input_image["image"].convert("RGB").size

        if task != "image-outpainting":
            if size1 < size2:
                input_image["image"] = input_image["image"].convert("RGB").resize((640, int(size2 / size1 * 640)))
            else:
                input_image["image"] = input_image["image"].convert("RGB").resize((int(size1 / size2 * 640), 640))
        else:
            if size1 < size2:
                input_image["image"] = input_image["image"].convert("RGB").resize((512, int(size2 / size1 * 512)))
            else:
                input_image["image"] = input_image["image"].convert("RGB").resize((int(size1 / size2 * 512), 512))

        if vertical_expansion_ratio is not None and horizontal_expansion_ratio is not None:
            o_W, o_H = input_image["image"].convert("RGB").size
            c_W = int(horizontal_expansion_ratio * o_W)
            c_H = int(vertical_expansion_ratio * o_H)

            expand_img = np.ones((c_H, c_W, 3), dtype=np.uint8) * 127
            original_img = np.array(input_image["image"])
            expand_img[
                int((c_H - o_H) / 2.0) : int((c_H - o_H) / 2.0) + o_H,
                int((c_W - o_W) / 2.0) : int((c_W - o_W) / 2.0) + o_W,
                :,
            ] = original_img

            blurry_gap = 10

            expand_mask = np.ones((c_H, c_W, 3), dtype=np.uint8) * 255
            if vertical_expansion_ratio == 1 and horizontal_expansion_ratio != 1:
                expand_mask[
                    int((c_H - o_H) / 2.0) : int((c_H - o_H) / 2.0) + o_H,
                    int((c_W - o_W) / 2.0) + blurry_gap : int((c_W - o_W) / 2.0) + o_W - blurry_gap,
                    :,
                ] = 0
            elif vertical_expansion_ratio != 1 and horizontal_expansion_ratio != 1:
                expand_mask[
                    int((c_H - o_H) / 2.0) + blurry_gap : int((c_H - o_H) / 2.0) + o_H - blurry_gap,
                    int((c_W - o_W) / 2.0) + blurry_gap : int((c_W - o_W) / 2.0) + o_W - blurry_gap,
                    :,
                ] = 0
            elif vertical_expansion_ratio != 1 and horizontal_expansion_ratio == 1:
                expand_mask[
                    int((c_H - o_H) / 2.0) + blurry_gap : int((c_H - o_H) / 2.0) + o_H - blurry_gap,
                    int((c_W - o_W) / 2.0) : int((c_W - o_W) / 2.0) + o_W,
                    :,
                ] = 0

            input_image["image"] = Image.fromarray(expand_img)
            input_image["mask"] = Image.fromarray(expand_mask)

        if self.version != "ppt-v1":
            if task == "image-outpainting":
                prompt = prompt + " empty scene"
            if task == "object-removal":
                prompt = prompt + " empty scene blur"
        print("Get prompt")
        print(prompt)

        promptA, promptB, negative_promptA, negative_promptB = add_task(prompt, negative_prompt, task, self.version)
        print(promptA, promptB, negative_promptA, negative_promptB)

        img = np.array(input_image["image"].convert("RGB"))
        W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
        H = int(np.shape(img)[1] - np.shape(img)[1] % 8)
        input_image["image"] = input_image["image"].resize((H, W))
        input_image["mask"] = input_image["mask"].resize((H, W))
        set_seed(seed)

        if self.version == "ppt-v1":
            # for sd-inpainting based method
            result = self.pipe(
                promptA=promptA,
                promptB=promptB,
                tradoff=fitting_degree,
                tradoff_nag=fitting_degree,
                negative_promptA=negative_promptA,
                negative_promptB=negative_promptB,
                image=input_image["image"].convert("RGB"),
                mask=input_image["mask"].convert("RGB"),
                width=H,
                height=W,
                guidance_scale=scale,
                num_inference_steps=ddim_steps,
            ).images[0]
        else:
            # for brushnet-based method
            np_inpimg = np.array(input_image["image"])
            np_inmask = np.array(input_image["mask"]) / 255.0
            np_inpimg = np_inpimg * (1 - np_inmask)
            input_image["image"] = Image.fromarray(np_inpimg.astype(np.uint8)).convert("RGB")
            result = self.pipe(
                promptA=promptA,
                promptB=promptB,
                promptU=prompt,
                tradoff=fitting_degree,
                tradoff_nag=fitting_degree,
                image=input_image["image"].convert("RGB"),
                mask=input_image["mask"].convert("RGB"),
                num_inference_steps=ddim_steps,
                generator=torch.Generator(device=device).manual_seed(seed),
                brushnet_conditioning_scale=1.0,
                negative_promptA=negative_promptA,
                negative_promptB=negative_promptB,
                negative_promptU=negative_prompt,
                guidance_scale=scale,
                width=H,
                height=W,
            ).images[0]

        mask_np = np.array(input_image["mask"].convert("RGB"))
        red = np.array(result).astype("float") * 1
        red[:, :, 0] = 180.0
        red[:, :, 2] = 0
        red[:, :, 1] = 0
        result_m = np.array(result)
        result_m = Image.fromarray(
            (
                result_m.astype("float") * (1 - mask_np.astype("float") / 512.0)
                + mask_np.astype("float") / 512.0 * red
            ).astype("uint8")
        )
        m_img = input_image["mask"].convert("RGB").filter(ImageFilter.GaussianBlur(radius=3))
        m_img = np.asarray(m_img) / 255.0
        img_np = np.asarray(input_image["image"].convert("RGB")) / 255.0
        ours_np = np.asarray(result) / 255.0
        ours_np = ours_np * m_img + (1 - m_img) * img_np
        dict_res = [input_image["mask"].convert("RGB"), result_m]

        # result_paste = Image.fromarray(np.uint8(ours_np * 255))
        # dict_out = [input_image["image"].convert("RGB"), result_paste]
        dict_out = [result]
        return dict_out, dict_res


    def predict_objr_batch(
        self,
        input_images,
        input_masks,
        prompt,
        promptA,
        promptB,
        negative_promptA,
        negative_promptB,
        fitting_degree,
        ddim_steps,
        scale,
        seed,
        negative_prompt,
        H,
        W,
        vertical_expansion_ratio,
        horizontal_expansion_ratio,
        device
    ):
        
        set_seed(seed)

        if self.version == "ppt-v1":
            # for sd-inpainting based method
            result = self.pipe(
                promptA=promptA,
                promptB=promptB,
                tradoff=fitting_degree,
                tradoff_nag=fitting_degree,
                negative_promptA=negative_promptA,
                negative_promptB=negative_promptB,
                image=input_images,
                mask=input_masks,
                width=H,
                height=W,
                guidance_scale=scale,
                num_inference_steps=ddim_steps,
            ).images[0]
        else:
            # for brushnet-based method
            # np_inpimg = input_images
            # np_inmask = input_masks / 255.0
            # np_inpimg = np_inpimg * (1 - np_inmask)
            # input_image["image"] = Image.fromarray(np_inpimg.astype(np.uint8)).convert("RGB")
            # print(input_images.shape)
            # exit(0)
            # print(input_images.shape)
            # print(input_masks.shape)
            # exit(0)
            result = self.pipe(
                promptA=promptA,
                promptB=promptB,
                promptU=prompt,
                tradoff=fitting_degree,
                tradoff_nag=fitting_degree,
                image=input_images,
                mask=input_masks,
                num_inference_steps=ddim_steps,
                generator=torch.Generator("cuda").manual_seed(seed),
                brushnet_conditioning_scale=1.0,
                negative_promptA=negative_promptA,
                negative_promptB=negative_promptB,
                negative_promptU=negative_prompt,
                guidance_scale=scale,
                width=H,
                height=W,
                device=device
            ).images[0]

        # mask_np = np.array(input_image["mask"].convert("RGB"))
        # red = np.array(result).astype("float") * 1
        # red[:, :, 0] = 180.0
        # red[:, :, 2] = 0
        # red[:, :, 1] = 0

        # result_paste = Image.fromarray(np.uint8(ours_np * 255))
        # dict_out = [input_image["image"].convert("RGB"), result_paste]
        dict_out = [result]
        return dict_out

    def predict_controlnet(
        self,
        input_image,
        input_control_image,
        control_type,
        prompt,
        ddim_steps,
        scale,
        seed,
        negative_prompt,
        controlnet_conditioning_scale,
    ):
        promptA = prompt + " P_obj"
        promptB = prompt + " P_obj"
        negative_promptA = negative_prompt
        negative_promptB = negative_prompt
        size1, size2 = input_image["image"].convert("RGB").size

        if size1 < size2:
            input_image["image"] = input_image["image"].convert("RGB").resize((640, int(size2 / size1 * 640)))
        else:
            input_image["image"] = input_image["image"].convert("RGB").resize((int(size1 / size2 * 640), 640))
        img = np.array(input_image["image"].convert("RGB"))
        W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
        H = int(np.shape(img)[1] - np.shape(img)[1] % 8)
        input_image["image"] = input_image["image"].resize((H, W))
        input_image["mask"] = input_image["mask"].resize((H, W))

        if control_type != self.current_control:
            self.load_controlnet(control_type)
        controlnet_image = input_control_image
        if control_type == "canny":
            controlnet_image = controlnet_image.resize((H, W))
            controlnet_image = np.array(controlnet_image)
            controlnet_image = cv2.Canny(controlnet_image, 100, 200)
            controlnet_image = controlnet_image[:, :, None]
            controlnet_image = np.concatenate([controlnet_image, controlnet_image, controlnet_image], axis=2)
            controlnet_image = Image.fromarray(controlnet_image)
        elif control_type == "pose":
            controlnet_image = self.openpose(controlnet_image)
        elif control_type == "depth":
            controlnet_image = controlnet_image.resize((H, W))
            controlnet_image = self.get_depth_map(controlnet_image)
        else:
            controlnet_image = self.hed(controlnet_image)

        mask_np = np.array(input_image["mask"].convert("RGB"))
        controlnet_image = controlnet_image.resize((H, W))
        set_seed(seed)
        result = self.control_pipe(
            promptA=promptB,
            promptB=promptA,
            tradoff=1.0,
            tradoff_nag=1.0,
            negative_promptA=negative_promptA,
            negative_promptB=negative_promptB,
            image=input_image["image"].convert("RGB"),
            mask=input_image["mask"].convert("RGB"),
            control_image=controlnet_image,
            width=H,
            height=W,
            guidance_scale=scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=ddim_steps,
        ).images[0]
        red = np.array(result).astype("float") * 1
        red[:, :, 0] = 180.0
        red[:, :, 2] = 0
        red[:, :, 1] = 0
        result_m = np.array(result)
        result_m = Image.fromarray(
            (
                result_m.astype("float") * (1 - mask_np.astype("float") / 512.0)
                + mask_np.astype("float") / 512.0 * red
            ).astype("uint8")
        )

        mask_np = np.array(input_image["mask"].convert("RGB"))
        m_img = input_image["mask"].convert("RGB").filter(ImageFilter.GaussianBlur(radius=4))
        m_img = np.asarray(m_img) / 255.0
        img_np = np.asarray(input_image["image"].convert("RGB")) / 255.0
        ours_np = np.asarray(result) / 255.0
        ours_np = ours_np * m_img + (1 - m_img) * img_np
        result_paste = Image.fromarray(np.uint8(ours_np * 255))
        return [input_image["image"].convert("RGB"), result_paste], [controlnet_image, result_m]

    def infer(
        self,
        input_image,
        text_guided_prompt,
        text_guided_negative_prompt,
        shape_guided_prompt,
        shape_guided_negative_prompt,
        fitting_degree,
        ddim_steps,
        scale,
        seed,
        task,
        vertical_expansion_ratio,
        horizontal_expansion_ratio,
        outpaint_prompt,
        outpaint_negative_prompt,
        removal_prompt,
        removal_negative_prompt,
        enable_control=False,
        input_control_image=None,
        control_type="canny",
        controlnet_conditioning_scale=None,
    ):
        if task == "text-guided":
            prompt = text_guided_prompt
            negative_prompt = text_guided_negative_prompt
        elif task == "shape-guided":
            prompt = shape_guided_prompt
            negative_prompt = shape_guided_negative_prompt
        elif task == "object-removal":
            prompt = removal_prompt
            negative_prompt = removal_negative_prompt
        elif task == "image-outpainting":
            prompt = outpaint_prompt
            negative_prompt = outpaint_negative_prompt
            return self.predict(
                input_image,
                prompt,
                fitting_degree,
                ddim_steps,
                scale,
                seed,
                negative_prompt,
                task,
                vertical_expansion_ratio,
                horizontal_expansion_ratio,
            )
        else:
            task = "text-guided"
            prompt = text_guided_prompt
            negative_prompt = text_guided_negative_prompt

        # currently, we only support controlnet in PowerPaint-v1
        if self.version == "ppt-v1" and enable_control and task == "text-guided":
            return self.predict_controlnet(
                input_image,
                input_control_image,
                control_type,
                prompt,
                ddim_steps,
                scale,
                seed,
                negative_prompt,
                controlnet_conditioning_scale,
            )
        else:
            return self.predict(
                input_image, prompt, fitting_degree, ddim_steps, scale, seed, negative_prompt, task, None, None
            )

    def infer_objr(
            self,
            input_image,
            removal_prompt,
            removal_negative_prompt,
            ddim_steps,
            scale,
            seed,
        ):
            task = "object-removal"
            prompt = removal_prompt
            negative_prompt = removal_negative_prompt
            fitting_degree = 1.0  # Default/fixed since not relevant for object-removal

            input_image
            input_image['image'].save("input.png")
            input_image['mask'].save("mask.png")
            output= self.predict(
                input_image,
                prompt,
                fitting_degree,
                ddim_steps,
                scale,
                seed,
                negative_prompt,
                task,
                None,  # vertical_expansion_ratio
                None,  # horizontal_expansion_ratio
            )
            print(output)
            output[0][0].save("test1.png")
            output[1][0].save("test2.png")
            output[1][1].save("test3.png")
            return output


class PowerPaintAugmentedDataset(Dataset):
    def __init__(self, metadata_csv, image_dir, mask_dir,
                 task="object-removal", version="ppt-v1",
                 vertical_expansion_ratio=None, horizontal_expansion_ratio=None, resolution=840):
        """
        Args:
            metadata_csv (str): CSV file with 'image_path' and 'mask_path' columns.
            image_dir (str): Directory containing the image files.
            mask_dir (str): Directory containing the mask files.
        """
        self.df = pd.read_csv(metadata_csv)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.task = task
        self.version = version
        self.vertical_expansion_ratio = vertical_expansion_ratio
        self.horizontal_expansion_ratio = horizontal_expansion_ratio
        self.resolution=resolution

        # Prompt construction
        # prompt = "remove object"
        # negative_prompt = "bad quality"
        prompt=""
        negative_prompt=""
        if self.version != "ppt-v1":
            if self.task == "image-outpainting":
                prompt += " empty scene"
            elif self.task == "object-removal":
                prompt += " empty scene blur"
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.promptA, self.promptB, self.negative_promptA, self.negative_promptB = add_task(prompt, negative_prompt, self.task, self.version)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row["image_path"])
        mask_path = os.path.join(self.mask_dir, row["mask_path"])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Step 1: Resize image by aspect ratio
        # size1, size2 = image.size
        # if size1 < size2:
        #     image = image.resize((640, int(size2 / size1 * 640)))
        #     mask = mask.resize((640, int(size2 / size1 * 640)))
        # else:
        #     image = image.resize((int(size1 / size2 * 512), 512))
        #     mask = mask.resize((int(size1 / size2 * 512), 512))
        image = image.resize((self.resolution, self.resolution))
        mask = mask.resize((self.resolution, self.resolution), Image.LANCZOS)

        

        # Step 2: Optional outpainting-style expansion
        if self.vertical_expansion_ratio is not None and self.horizontal_expansion_ratio is not None:
            o_W, o_H = image.size
            c_W = int(self.horizontal_expansion_ratio * o_W)
            c_H = int(self.vertical_expansion_ratio * o_H)

            # Expanded image
            expand_img = np.ones((c_H, c_W, 3), dtype=np.uint8) * 127
            original_img = np.array(image)
            expand_img[
                int((c_H - o_H) / 2):int((c_H - o_H) / 2) + o_H,
                int((c_W - o_W) / 2):int((c_W - o_W) / 2) + o_W,
                :
            ] = original_img

            # Expanded mask
            blurry_gap = 10
            expand_mask = np.ones((c_H, c_W, 3), dtype=np.uint8) * 255
            if self.vertical_expansion_ratio == 1 and self.horizontal_expansion_ratio != 1:
                expand_mask[
                    int((c_H - o_H) / 2):int((c_H - o_H) / 2) + o_H,
                    int((c_W - o_W) / 2) + blurry_gap:int((c_W - o_W) / 2) + o_W - blurry_gap,
                    :
                ] = 0
            elif self.vertical_expansion_ratio != 1 and self.horizontal_expansion_ratio != 1:
                expand_mask[
                    int((c_H - o_H) / 2) + blurry_gap:int((c_H - o_H) / 2) + o_H - blurry_gap,
                    int((c_W - o_W) / 2) + blurry_gap:int((c_W - o_W) / 2) + o_W - blurry_gap,
                    :
                ] = 0
            elif self.vertical_expansion_ratio != 1 and self.horizontal_expansion_ratio == 1:
                expand_mask[
                    int((c_H - o_H) / 2) + blurry_gap:int((c_H - o_H) / 2) + o_H - blurry_gap,
                    int((c_W - o_W) / 2):int((c_W - o_W) / 2) + o_W,
                    :
                ] = 0

            image = Image.fromarray(expand_img)
            mask = Image.fromarray(expand_mask)

           


        # Step 3: Resize to nearest multiple of 8
        W, H = image.size
        H = H - (H % 8)
        W = W - (W % 8)
        image = image.resize((W, H))
        mask = mask.resize((W, H), Image.LANCZOS)
        mask_np = np.array(mask.convert("L"))
        #convert bounding box
        # Step 2: Create binary mask (object = 1, background = 0)
        binary_mask = (mask_np >= 180).astype(np.uint8)

        # Step 3: Find bounding box of the object
        ys, xs = np.where(binary_mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            print("No object found in mask.")
            # Optionally: create all-black mask
            new_mask = Image.fromarray(np.zeros_like(mask_np, dtype=np.uint8))
        else:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            # Step 4: Create new binary mask (all black)
            new_mask_np = np.zeros_like(mask_np, dtype=np.uint8)

            # Step 5: Fill the bounding box region with white
            new_mask_np[y_min:y_max+1, x_min:x_max+1] = 255

            # Step 6: Convert back to PIL.Image
            new_mask = Image.fromarray(new_mask_np)

        # Now `new_mask` replaces the original mask
        mask = new_mask

        if self.version != "ppt-v1":
            # np_inpimg = np.array(input_image["image"])
            # np_inmask = np.array(input_image["mask"]) / 255.0
            # np_inpimg = np_inpimg * (1 - np_inmask)
            # Convert image and mask
            np_image = np.array(image).astype(np.uint8)
            np_mask = np.array(mask).astype(np.float32)   # Normalize to [0, 1]
            norm_np_mask = np_mask /255.0

            # Threshold mask: < 0.7 → 0, ≥ 0.7 → 1
            binary_mask = (np_mask >= 0.7).astype(np.float32)

            # Expand binary mask to 3 channels for masking the image
            np_mask_3ch = np.repeat(binary_mask[:, :, None], 3, axis=2)

            # Apply mask to image (zero out masked regions)
            np_image = np_image * (1 - np_mask_3ch)

            # Convert original mask to [-1, 1] range (after thresholding)
            # np_mask = np_mask_3ch * 2.0 - 1.0  # from {0, 1} → {-1, 1}

        else:
            np_image = np.array(image).astype(np.uint8)
            np_mask = np.array(mask).astype(np.float32)   # normalize
            norm_np_mask = np_mask /255.0
            # Threshold mask: < 0.7 → 0, ≥ 0.7 → 1
            # Threshold mask: < 0.7 → 0, ≥ 0.7 → 1
            binary_mask = (norm_np_mask >= 0.7).astype(np.float32)
            # Expand binary mask to 3 channels for masking the image
            np_mask_3ch = np.repeat(binary_mask[:, :, None], 3, axis=2)
            # Convert original mask to [-1, 1] range (after thresholding)
            # np_mask = np_mask_3ch * 2.0 - 1.0  # from {0, 1} → {-1, 1}
           

        # np_image = np.transpose(np_image, (2, 0, 1))
        # # np_mask = np.transpose(np_mask, (2,0,1))
        # np_mask = np.transpose(np_mask, (2, 0, 1))
        # np_mask = np_mask[:, :, 0]
        np_mask = np.repeat(np_mask[:, :, None], 3, axis=2)
        image = TF.to_tensor(Image.fromarray(np_image.astype("uint8")))
        mask = TF.to_tensor(Image.fromarray(np_mask.astype("uint8")))

        image = TF.normalize(image, mean=[0.5]*3, std=[0.5]*3)
        
    
        return {
            "image": image,
            "mask": mask,
            "prompt": self.prompt,
            "promptA": self.promptA,
            "promptB": self.promptB,
            "negative_promptA": self.negative_promptA,
            "negative_promptB": self.negative_promptB,
            "negative_prompt": self.negative_prompt,
            "filename": os.path.basename(image_path),
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta",  type=str, default="/home/ubuntu/data/ldm_random_crop.myntra40k_amz67k.cleaned_no_whbg.biref.v1.1.rm_whitemargin/record_v1.csv" )
    parser.add_argument("--image_dir", type=str, default="/home/ubuntu/data/ldm_random_crop.myntra40k_amz67k.cleaned_no_whbg.biref.v1.1.rm_whitemargin/images")
    parser.add_argument("--mask_dir", type=str, default="/home/ubuntu/data/ldm_random_crop.myntra40k_amz67k.cleaned_no_whbg.biref.v1.1.rm_whitemargin/images")
    parser.add_argument("--checkpoint_dir", default="./checkpoints/ppt-v2_cn")
    parser.add_argument("--version", default="ppt-v2")
    parser.add_argument("--weight_dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--output_dir", default="/home/ubuntu/data/ldm_random_crop.myntra40k_amz67k.cleaned_no_whbg.biref.v1.1.rm_whitemargin/images_bg_full/")
    parser.add_argument("--ddim_steps", type=int, default=45)
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=840)
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision="fp16")

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float16 if args.weight_dtype == "float16" else torch.float32

    # Initialize model/controller
    controller = PowerPaintController(
        weight_dtype=weight_dtype,
        checkpoint_dir=args.checkpoint_dir,
        local_files_only=args.local_files_only,
        version=args.version,
        device=accelerator.device
    )

    # Prepare dataset
    dataset = PowerPaintAugmentedDataset(
        metadata_csv=args.meta,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        version=args.version,
        resolution=args.resolution,
        vertical_expansion_ratio=None,
        horizontal_expansion_ratio=None,
    )

    sampler = DistributedSampler(dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index, shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    # Prepare everything with accelerator
    # dataloader = accelerator.prepare(dataloader)
    controller.pipe = accelerator.prepare(controller.pipe)
    if hasattr(controller, "control_pipe"):
        controller.control_pipe = accelerator.prepare(controller.control_pipe)
    controller.pipe = controller.pipe.to(accelerator.device)
    controller.pipe.unet = controller.pipe.unet.to(accelerator.device)
    controller.pipe.brushnet = controller.pipe.brushnet.to(accelerator.device)
    controller.pipe.text_encoder_brushnet = controller.pipe.text_encoder_brushnet.to(accelerator.device)


    print(f"Model device: {next(controller.pipe.unet.parameters()).device}")
    # Inference
    print("Starting inference...")
    count = 0
    for batch in dataloader:
        input_images = batch["image"].to(accelerator.device)
        input_masks = batch["mask"].to(accelerator.device)
        count += 1
        if accelerator.is_main_process:
            # Print progress only on the main process
            if count % 10 == 0:
                print(f"Processed {count} batches over {len(dataloader)} so far.")
            
        
        #____________________________________________
        with torch.no_grad(), accelerator.autocast():
            outputs = controller.predict_objr_batch(
                input_images=input_images,
                input_masks=input_masks,
                prompt=batch["prompt"],
                promptA=batch["promptA"],
                promptB=batch["promptB"],
                negative_promptA=batch["negative_promptA"],
                negative_promptB=batch["negative_promptB"],
                fitting_degree=1.0,
                ddim_steps=args.ddim_steps,
                scale=args.scale,
                seed=args.seed,
                negative_prompt=batch["negative_prompt"],
                H=args.resolution,
                W=args.resolution,
                vertical_expansion_ratio=None,
                horizontal_expansion_ratio=None,
                device=accelerator.device
            )

        
            for i, img in enumerate(outputs):
                filename = os.path.splitext(batch["filename"][i])[0] + "_out.png"
                save_path = os.path.join(args.output_dir, filename)
                img.save(save_path)
        #________________________________________________
        if count % 10 == 0:
            print(f"Processed {count} batches over {len(dataloader)} in rank {accelerator.process_index} so far.")
        accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--meta",  type=str, default="/home/ubuntu/data/ldm_random_crop.myntra40k_amz67k.cleaned_no_whbg.biref.v1.1.rm_whitemargin/record_v1.csv" )
#     parser.add_argument("--image_dir", type=str, default="/home/ubuntu/data/ldm_random_crop.myntra40k_amz67k.cleaned_no_whbg.biref.v1.1.rm_whitemargin/images")
#     parser.add_argument("--mask_dir", type=str, default="/home/ubuntu/data/ldm_random_crop.myntra40k_amz67k.cleaned_no_whbg.biref.v1.1.rm_whitemargin/images")
#     parser.add_argument("--checkpoint_dir", default="./checkpoints/ppt-v2_cn")
#     parser.add_argument("--version", default="ppt-v2")
#     parser.add_argument("--weight_dtype", default="float16")
#     parser.add_argument("--vertical_expansion_ratio", type=float, default=None)
#     parser.add_argument("--horizontal_expansion_ratio", type=float, default=None)
#     parser.add_argument("--output_dir", default="/home/ubuntu/data/ldm_random_crop.myntra40k_amz67k.cleaned_no_whbg.biref.v1.1.rm_whitemargin/images_bg/")
#     parser.add_argument("--ddim_steps", type=int, default=45)
#     parser.add_argument("--scale", type=float, default=10.0)
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--batch_size", type=int, default=1)
#     parser.add_argument("--local_files_only", action="store_true", help="enable it to use cached files without requesting from the hub")
#     parser.add_argument("--resolution", type=int, default=840)
#     args = parser.parse_args()

#     accelerator = Accelerator()
#     device = accelerator.device

#     weight_dtype = torch.float16 if args.weight_dtype == "float16" else torch.float32
#     controller = PowerPaintController(weight_dtype, args.checkpoint_dir, local_files_only=False, version=args.version)
#     controller.pipe.to(device)
#     controller.set_eval_mode()
#     dataset = PowerPaintAugmentedDataset(
#         metadata_csv=args.meta,
#         image_dir=args.image_dir,
#         mask_dir=args.mask_dir,
#         version=args.version,
#         vertical_expansion_ratio=args.vertical_expansion_ratio,
#         horizontal_expansion_ratio=args.horizontal_expansion_ratio,
#         resolution = args.resolution,
#     )

#     # def collate_fn(batch):
#     #     return {
#     #         "image": [item["image"] for item in batch],
#     #         "mask": [item["mask"] for item in batch],
#     #         "prompt": [item["prompt"] for item in batch],
#     #         "promptA": [item["promptA"] for item in batch],
#     #         "promptB": [item["promptB"] for item in batch],
#     #         "negative_promptA": [item["negative_promptA"] for item in batch],
#     #         "negative_promptB": [item["negative_promptB"] for item in batch],
#     #         "filename": [item["filename"] for item in batch],
#     #     }
#     # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
#     dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
#     dataloader = accelerator.prepare(dataloader)

#     os.makedirs(args.output_dir, exist_ok=True)

#     for batch in dataloader:
#         input_images = batch["image"]
        
#         input_masks = batch["mask"]
#         # save_image(input_images[0], "debug_first_image.png")
#         # save_image(input_masks[0], "debug_first_mask.png")
#         # exit(0)
#         prompt = batch["prompt"]
#         promptA = batch["promptA"]
#         promptB = batch["promptB"]

#         negative_promptA = batch["negative_promptA"]
#         negative_promptB = batch["negative_promptB"]
#         negative_prompt = batch["negative_prompt"]
#         # negative_prompt = batch["negative_prompt"][0]
#         with torch.no_grad():
#             with torch.cuda.amp.autocast(enabled=accelerator.mixed_precision == "fp16"):
                
#                 outputs = controller.predict_objr_batch(
#                     input_images=input_images,
#                     input_masks=input_masks,
#                     prompt=prompt,
#                     promptA=promptA,
#                     promptB=promptB,
#                     negative_promptA=negative_promptA,
#                     negative_promptB=negative_promptB,
#                     fitting_degree=1.0,
#                     ddim_steps=args.ddim_steps,
#                     scale=args.scale,
#                     seed=args.seed,
#                     negative_prompt=negative_prompt,
#                     H = args.resolution,
#                     W = args.resolution,
#                     vertical_expansion_ratio=None,
#                     horizontal_expansion_ratio=None,
#                 )

        
        
#         for i, out_img in enumerate(outputs):
#             filename = os.path.splitext(batch["filename"][i])[0] + "_out.png"
#             save_path = os.path.join(args.output_dir, filename)

#             # if accelerator.is_local_main_process:
#             out_img.save(save_path)

#     accelerator.wait_for_everyone()


