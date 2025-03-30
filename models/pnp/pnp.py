import torch
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
    StableDiffusionXLPipeline,
)
import numpy as np
from PIL import Image
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
from models.p2p.inversion import (
    DirectInversion,
    NullInversion,
)  # , NegativePromptInversion
import torchvision.transforms as T
from compel import Compel, ReturnedEmbeddingsType

from utils.utils import txt_draw, load_512, latent2image


def get_timesteps(scheduler, num_inference_steps, strength, device):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]

    return timesteps, num_inference_steps - t_start


class Preprocess(nn.Module):
    def __init__(self, device, model_key, num_ddim_steps):
        super().__init__()

        self.device = device
        self.use_depth = False

        # Create model
        print(f"[Preprocess] loading stable diffusion...")
        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        self.model = StableDiffusionXLPipeline.from_pretrained(
            model_key, scheduler=self.scheduler
        ).to("cuda")
        self.model.enable_xformers_memory_efficient_attention()
        self.model.scheduler.set_timesteps(num_ddim_steps)

        print(f"[Preprocess] loaded stable diffusion!")

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, device="cuda"):
        print("✅ get_text_embeds() - %s / %s" % (prompt, negative_prompt))

        compel = Compel(
            tokenizer=[self.model.tokenizer, self.model.tokenizer_2],
            text_encoder=[self.model.text_encoder, self.model.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
        )

        prompt_embeds, pooled_prompt_embeds = compel(prompt)
        negative_prompt_embeds, negative_pooled_prompt_embeds = compel(negative_prompt)

        vae_scale_factor = 2 ** (len(self.model.vae.config.block_out_channels) - 1)
        default_sample_size = self.model.unet.config.sample_size

        height = default_sample_size * vae_scale_factor
        width = default_sample_size * vae_scale_factor

        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.model.unet.config.addition_time_embed_dim * len(add_time_ids)
            + self.model.text_encoder_2.config.projection_dim
        )
        expected_add_embed_dim = self.model.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=self.model.unet.dtype).to(
            self.device
        )
        batch_size = prompt_embeds.shape[0]
        add_time_ids = add_time_ids.repeat(batch_size, 1)

        context = torch.cat([negative_prompt_embeds, prompt_embeds])
        context_p = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])

        add_time_ids = torch.cat([add_time_ids, add_time_ids])

        return context, context_p, add_time_ids

    @torch.no_grad()
    def decode_latents(self, latents):
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            latents = 1 / 0.13025 * latents
            imgs = self.model.vae.decode(latents).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def load_img(self, image_path):
        image_pil = T.Resize([512, 512])(Image.open(image_path).convert("RGB"))
        image = T.ToTensor()(image_pil).unsqueeze(0).to(self.device)
        return image

    @torch.no_grad()
    def encode_imgs(self, imgs):
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            imgs = 2 * imgs - 1
            posterior = self.model.vae.encode(imgs).latent_dist
            latents = posterior.mean * 0.13025
        return latents

    @torch.no_grad()
    def ddim_inversion(self, cond, cond_p, add_time, latent):
        print("✅ ddim_inversion()")
        latent_list = [latent]
        timesteps = reversed(self.model.scheduler.timesteps)
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            for i, t in enumerate(timesteps):
                cond_batch = cond.repeat(latent.shape[0], 1, 1)
                # 배치 처리 전/후 형상 변함 없음
                # cond_p, add_time은 배치 생략

                t = t.int().item()
                alpha_prod_t = self.model.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.model.scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0
                    else self.model.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t**0.5
                mu_prev = alpha_prod_t_prev**0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                added_cond_kwargs = {
                    "text_embeds": cond_p,
                    "time_ids": add_time,
                }
                eps = self.model.unet(
                    latent,
                    t,
                    encoder_hidden_states=cond_batch,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

                pred_x0 = (latent - sigma_prev * eps) / mu_prev
                latent = mu * pred_x0 + sigma * eps
                latent_list.append(latent)
        return latent_list

    @torch.no_grad()
    def ddim_sample(self, x, cond, cond_p, add_time):
        print("✅ ddim_sample()")
        timesteps = self.model.scheduler.timesteps
        latent_list = []
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            for i, t in enumerate(timesteps):
                cond_batch = cond.repeat(x.shape[0], 1, 1)

                alpha_prod_t = self.model.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.model.scheduler.alphas_cumprod[timesteps[i + 1]]
                    if i < len(timesteps) - 1
                    else self.model.scheduler.final_alpha_cumprod
                )
                mu = alpha_prod_t**0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                mu_prev = alpha_prod_t_prev**0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                added_cond_kwargs = {
                    "text_embeds": cond_p,
                    "time_ids": add_time,
                }
                eps = self.model.unet(
                    x,
                    t,
                    encoder_hidden_states=cond_batch,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

                pred_x0 = (x - sigma * eps) / mu
                x = mu_prev * pred_x0 + sigma_prev * eps
                latent_list.append(x)
        return latent_list

    def prev_step(self, model_output, timestep: int, sample):
        prev_timestep = (
            timestep
            - self.model.scheduler.config.num_train_timesteps
            // self.model.scheduler.num_inference_steps
        )
        alpha_prod_t = self.model.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.model.scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.model.scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (
            sample - beta_prod_t**0.5 * model_output
        ) / alpha_prod_t**0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = (
            alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction
        )

        difference_scale_pred_original_sample = -(beta_prod_t**0.5) / alpha_prod_t**0.5
        difference_scale_pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5
        difference_scale = (
            alpha_prod_t_prev**0.5 * difference_scale_pred_original_sample
            + difference_scale_pred_sample_direction
        )

        return prev_sample, difference_scale

    def get_noise_pred_single(self, latents, t, context):
        print("✅ get_noise_pred_single()")
        # print("latent shape",latents.shape)
        # print("context shape",context.shape)
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context).sample
        return noise_pred

    @torch.no_grad()
    def extract_latents(
        self, num_steps, data_path, inversion_prompt="", guidance_scale=7.5
    ):
        print("✅ extract_latents()")
        self.model.scheduler.set_timesteps(num_steps)

        cond, cond_p, add_time = self.get_text_embeds(inversion_prompt, "")
        cond = cond[1].unsqueeze(0)
        cond_p = cond_p[1].unsqueeze(0)
        add_time = add_time[1].unsqueeze(0)

        image = self.load_img(data_path)
        latent = self.encode_imgs(image)

        inverted_x = self.ddim_inversion(
            cond, cond_p, add_time, latent
        )  # X0, X1, ..., XT
        latent_reconstruction = self.ddim_sample(
            inverted_x[-1], cond, cond_p, add_time
        )  # XT', XT-1', ..., X0'
        rgb_reconstruction = self.decode_latents(latent_reconstruction[-1])
        latent_reconstruction.reverse()  # X0', X1', ..., XT'
        return inverted_x, rgb_reconstruction, latent_reconstruction


def register_time(model, t):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, "t", t)
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            if res >= len(model.unet.up_blocks) or not hasattr(
                model.unet.up_blocks[res], "attentions"
            ):
                continue

            module = (
                model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            )
            setattr(module, "t", t)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            if res >= len(model.unet.down_blocks) or not hasattr(
                model.unet.down_blocks[res], "attentions"
            ):
                continue

            module = (
                model.unet.down_blocks[res]
                .attentions[block]
                .transformer_blocks[0]
                .attn1
            )
            setattr(module, "t", t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, "t", t)


def register_attention_control_efficient(model, injection_schedule):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            print("✅ sa_forward()")
            batch_size, sequence_length, dim = x.shape
            h = self.heads

            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            if (
                not is_cross
                and self.injection_schedule is not None
                and (self.t in self.injection_schedule or self.t == 1000)
            ):
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)

                source_batch_size = int(q.shape[0] // 3)
                # inject unconditional
                q[source_batch_size : 2 * source_batch_size] = q[:source_batch_size]
                k[source_batch_size : 2 * source_batch_size] = k[:source_batch_size]
                # inject conditional
                q[2 * source_batch_size :] = q[:source_batch_size]
                k[2 * source_batch_size :] = k[:source_batch_size]

                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)
            else:
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)
                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)

            v = self.to_v(encoder_hidden_states)
            v = self.head_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)

            return to_out(out)

        return forward

    res_dict = {
        1: [1, 2],
        2: [0, 1, 2],
        3: [0, 1, 2],
    }  # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution

    for res in res_dict:
        for block in res_dict[res]:
            if res >= len(model.unet.up_blocks) or not hasattr(
                model.unet.up_blocks[res], "attentions"
            ):
                continue

            module = (
                model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            )
            module.forward = sa_forward(module)
            setattr(module, "injection_schedule", injection_schedule)


def register_conv_control_efficient(model, injection_schedule):
    def conv_forward(self):
        def forward(input_tensor, temb):
            print("✅ conv_forward()")
            hidden_states = input_tensor

            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)

            hidden_states = self.conv1(hidden_states)

            if temb is not None:
                temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            hidden_states = self.norm2(hidden_states)

            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)
            if self.injection_schedule is not None and (
                self.t in self.injection_schedule or self.t == 1000
            ):
                source_batch_size = int(hidden_states.shape[0] // 3)
                # inject unconditional
                hidden_states[source_batch_size : 2 * source_batch_size] = (
                    hidden_states[:source_batch_size]
                )
                # inject conditional
                hidden_states[2 * source_batch_size :] = hidden_states[
                    :source_batch_size
                ]

            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            return output_tensor

        return forward

    conv_module = model.unet.up_blocks[1].resnets[1]
    conv_module.forward = conv_forward(conv_module)
    setattr(conv_module, "injection_schedule", injection_schedule)


class PNP(nn.Module):
    def __init__(self, num_ddim_steps=50, device="cuda"):
        super().__init__()
        self.device = device
        model_key = "stabilityai/stable-diffusion-xl-base-1.0"

        # Create SD models
        print("Loading SD model")
        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_key, scheduler=self.scheduler
        ).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.scheduler.set_timesteps(num_ddim_steps, device=self.device)
        self.num_ddim_steps = num_ddim_steps

        self.toy_scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler"
        )
        self.toy_scheduler.set_timesteps(self.num_ddim_steps)
        timesteps_to_save, num_inference_steps = get_timesteps(
            self.toy_scheduler,
            num_inference_steps=self.num_ddim_steps,
            strength=1.0,
            device=self.device,
        )
        self.timesteps_to_save = timesteps_to_save
        self.num_inference_steps = num_inference_steps
        self.preprocessor = Preprocess(
            self.device, model_key=model_key, num_ddim_steps=num_ddim_steps
        )

        print("SD model loaded")

    def __call__(
        self,
        edit_method,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        image_size=[512, 512],
    ):
        if edit_method == "ddim+pnp":
            return self.edit_image_ddim_PnP(
                image_path, prompt_src, prompt_tar, guidance_scale, image_size
            )
        elif edit_method == "directinversion+pnp":
            return self.edit_image_directinversion_PnP(
                image_path, prompt_src, prompt_tar, guidance_scale, image_size
            )
        elif edit_method == "null-text-inversion+pnp":
            return self.edit_image_null_text_inversion_pnp(
                image_path, prompt_src, prompt_tar, guidance_scale, image_size
            )
        elif edit_method == "negative-prompt-inversion+pnp":
            return self.edit_image_negative_prompt_inversion_pnp(
                image_path, prompt_src, prompt_tar, guidance_scale, image_size
            )

        else:
            raise ValueError(f"edit method {edit_method} not supported")

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, batch_size=1):
        print("✅ get_text_embeds")
        # Tokenize text and get embeddings

        compel = Compel(
            tokenizer=[
                self.preprocessor.model.tokenizer,
                self.preprocessor.model.tokenizer_2,
            ],
            text_encoder=[
                self.preprocessor.model.text_encoder,
                self.preprocessor.model.text_encoder_2,
            ],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
        )

        prompt_embeds, pooled_prompt_embeds = compel(prompt)
        negative_prompt_embeds, negative_pooled_prompt_embeds = compel(negative_prompt)

        vae_scale_factor = 2 ** (
            len(self.preprocessor.model.vae.config.block_out_channels) - 1
        )
        default_sample_size = self.preprocessor.model.unet.config.sample_size

        height = default_sample_size * vae_scale_factor
        width = default_sample_size * vae_scale_factor

        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.preprocessor.model.unet.config.addition_time_embed_dim
            * len(add_time_ids)
            + self.preprocessor.model.text_encoder_2.config.projection_dim
        )
        expected_add_embed_dim = (
            self.preprocessor.model.unet.add_embedding.linear_1.in_features
        )

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor(
            [add_time_ids], dtype=self.preprocessor.model.unet.dtype
        ).to(self.device)
        batch_size = prompt_embeds.shape[0]
        add_time_ids = add_time_ids.repeat(batch_size, 1)

        context = torch.cat([negative_prompt_embeds, prompt_embeds])
        context_p = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])

        add_time_ids = torch.cat([add_time_ids, add_time_ids])

        return context, context_p, add_time_ids

    @torch.no_grad()
    def decode_latent(self, latent):
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            latent = 1 / 0.13025 * latent
            img = self.vae.decode(latent).sample
            img = (img / 2 + 0.5).clamp(0, 1)
        return img

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def get_data(self, image_path):
        # load image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((512, 512), resample=Image.Resampling.LANCZOS)
        image = T.ToTensor()(image).to(self.device)
        return image

    @torch.no_grad()
    def denoise_step(
        self, x, i, t, guidance_scale, noisy_latent, null_or_negative=False
    ):
        # register the time step and features in pnp injection modules
        latent_model_input = torch.cat(([noisy_latent] + [x] * 2))

        register_time(self, t.item())

        # compute text embeddings
        if null_or_negative:
            print("🌊 NULL-TEXT INVERSION")
            text_embed_input = torch.cat(
                [
                    self.pnp_guidance_embeds[i].expand(*self.text_embeds.shape),
                    self.text_embeds.chunk(2)[1],
                ],
                dim=0,
            )
            text_embed_input_p = torch.cat(
                [
                    self.pnp_guidance_embeds_p[i].expand(*self.text_embeds_p.shape),
                    self.text_embeds_p.chunk(2)[1],
                ],
                dim=0,
            )
            add_time_ids_input = torch.cat(
                [self.pnp_guidance_time_ids, self.add_time_ids], dim=0
            )
        else:
            text_embed_input = torch.cat(
                [self.pnp_guidance_embeds, self.text_embeds], dim=0
            )
            text_embed_input_p = torch.cat(
                [self.pnp_guidance_embeds_p, self.text_embeds_p], dim=0
            )
            add_time_ids_input = torch.cat(
                [self.pnp_guidance_time_ids, self.add_time_ids], dim=0
            )

        print("📦 self.pnp_guidance_time_ids =", self.pnp_guidance_time_ids.shape)
        print("📦 self.add_time_ids =", self.add_time_ids.shape)

        print("📦 latent_model_input =", latent_model_input.shape)
        print("📦 text_embed_input =", text_embed_input.shape)
        print("📦 text_embed_input_p =", text_embed_input_p.shape)
        print("📦 add_time_ids_input =", add_time_ids_input.shape)

        # apply the denoising network
        added_cond_kwargs = {
            "text_embeds": text_embed_input_p,
            "time_ids": add_time_ids_input,
        }
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embed_input,
            added_cond_kwargs=added_cond_kwargs,
        )["sample"]

        print("📦 noise_pred =", noise_pred.shape)

        # perform guidance
        _, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        # compute the denoising step with the reference model
        denoised_latent = self.scheduler.step(noise_pred, t, x)["prev_sample"]
        # if noise_loss is not None:
        #     denoised_latent = torch.concat((denoised_latent[:1]+noise_loss[:1],denoised_latent[1:]))
        return denoised_latent

    def init_pnp(self, conv_injection_t, qk_injection_t):
        self.qk_injectionum_ddim_steps = (
            self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        )
        self.conv_injectionum_ddim_steps = (
            self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        )
        register_attention_control_efficient(self, self.qk_injectionum_ddim_steps)
        register_conv_control_efficient(self, self.conv_injectionum_ddim_steps)

    def run_pnp(
        self,
        image_path,
        noisy_latent,
        target_prompt,
        guidance_scale=7.5,
        uncond_embeddings=None,
        uncond_embeddings_p=None,
        pnp_f_t=0.8,
        pnp_attn_t=0.5,
    ):
        print("✅ run_pnp")
        # load image
        self.image = self.get_data(image_path)
        self.eps = noisy_latent[-1]

        self.text_embeds, self.text_embeds_p, self.add_time_ids = self.get_text_embeds(
            target_prompt, "ugly, blurry, black, low res, unrealistic"
        )

        embeds, embeds_p, add_time_ids = self.get_text_embeds("", "")
        if uncond_embeddings is None:
            self.pnp_guidance_embeds = embeds.chunk(2)[0]
            self.pnp_guidance_embeds_p = embeds_p.chunk(2)[0]
            self.pnp_guidance_time_ids = add_time_ids.chunk(2)[0]
        else:
            print(
                "🌊 UNCOND_EMBEDDING: ",
                len(uncond_embeddings),
                len(uncond_embeddings_p),
            )
            self.pnp_guidance_embeds = uncond_embeddings
            self.pnp_guidance_embeds_p = uncond_embeddings_p
            self.pnp_guidance_time_ids = add_time_ids.chunk(2)[0]

        pnp_f_t = int(self.num_ddim_steps * pnp_f_t)
        pnp_attn_t = int(self.num_ddim_steps * pnp_attn_t)
        self.init_pnp(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)
        if uncond_embeddings is None:
            edited_img = self.sample_loop(self.eps, guidance_scale, noisy_latent)
        else:
            edited_img = self.sample_loop(self.eps, guidance_scale, noisy_latent, True)
        return edited_img

    def sample_loop(self, x, guidance_scale, noisy_latent, null_or_negative=False):
        print("✅ sample_loop()")
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            for i, t in enumerate(self.scheduler.timesteps):
                if null_or_negative:
                    x = self.denoise_step(
                        x, i, t, guidance_scale, noisy_latent[-1 - i], null_or_negative
                    )
                else:
                    x = self.denoise_step(x, i, t, guidance_scale, noisy_latent[-1 - i])
            decoded_latent = self.decode_latent(x)

        return decoded_latent

    def edit_image_ddim_PnP(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        image_shape=[512, 512],
    ):
        torch.cuda.empty_cache()
        image_gt = load_512(image_path)
        _, rgb_reconstruction, latent_reconstruction = (
            self.preprocessor.extract_latents(
                data_path=image_path,
                num_steps=self.num_ddim_steps,
                inversion_prompt=prompt_src,
                guidance_scale=guidance_scale,
            )
        )

        edited_image = self.run_pnp(
            image_path, latent_reconstruction, prompt_tar, guidance_scale
        )

        image_instruct = txt_draw(
            f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}"
        )

        return Image.fromarray(
            np.concatenate(
                (
                    image_instruct,
                    image_gt,
                    np.uint8(
                        255
                        * np.array(
                            rgb_reconstruction[0].permute(1, 2, 0).cpu().detach()
                        )
                    ),
                    np.uint8(
                        255 * np.array(edited_image[0].permute(1, 2, 0).cpu().detach())
                    ),
                ),
                1,
            )
        )

    def edit_image_directinversion_PnP(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        image_shape=[512, 512],
    ):
        torch.cuda.empty_cache()
        image_gt = load_512(image_path)
        inverted_x, _, __ = self.preprocessor.extract_latents(
            data_path=image_path,
            num_steps=self.num_ddim_steps,
            inversion_prompt=prompt_src,
        )

        edited_image = self.run_pnp(image_path, inverted_x, prompt_tar, guidance_scale)

        image_instruct = txt_draw(
            f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}"
        )

        return Image.fromarray(
            np.concatenate(
                (
                    image_instruct,
                    image_gt,
                    np.uint8(
                        np.array(
                            latent2image(
                                model=self.vae, latents=inverted_x[1].to(self.vae.dtype)
                            )[0]
                        )
                    ),
                    np.uint8(
                        255 * np.array(edited_image[0].permute(1, 2, 0).cpu().detach())
                    ),
                ),
                1,
            )
        )

    def edit_image_null_text_inversion_pnp(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        image_shape=[512, 512],
    ):
        torch.cuda.empty_cache()
        image_gt = load_512(image_path)

        null_inversion = NullInversion(
            model=self.preprocessor.model, num_ddim_steps=self.num_ddim_steps
        )

        _, _, inverted_x, uncond_embeddings, uncond_embeddings_p = (
            null_inversion.invert(
                image_gt=image_gt, prompt=prompt_src, guidance_scale=guidance_scale
            )
        )

        edited_image = self.run_pnp(
            image_path,
            inverted_x,
            prompt_tar,
            guidance_scale,
            uncond_embeddings,
            uncond_embeddings_p,
        )

        image_instruct = txt_draw(
            f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}"
        )

        return Image.fromarray(
            np.concatenate(
                (
                    image_instruct,
                    image_gt,
                    np.uint8(
                        np.array(
                            latent2image(
                                model=self.vae, latents=inverted_x[1].to(self.vae.dtype)
                            )[0]
                        )
                    ),
                    np.uint8(
                        255 * np.array(edited_image[0].permute(1, 2, 0).cpu().detach())
                    ),
                ),
                1,
            )
        )

    def edit_image_negative_prompt_inversion_pnp(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        image_shape=[512, 512],
    ):
        torch.cuda.empty_cache()
        image_gt = load_512(image_path)

        negative_inversion = NegativePromptInversion(
            model=self.preprocessor.model, num_ddim_steps=self.num_ddim_steps
        )

        _, _, inverted_x, uncond_embeddings, uncond_embeddings_p = (
            negative_inversion.invert(image_gt=image_gt, prompt=prompt_src)
        )

        edited_image = self.run_pnp(
            image_path,
            inverted_x,
            prompt_tar,
            guidance_scale,
            uncond_embeddings,
            uncond_embeddings_p,
        )

        image_instruct = txt_draw(
            f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}"
        )

        return Image.fromarray(
            np.concatenate(
                (
                    image_instruct,
                    image_gt,
                    np.uint8(
                        np.array(
                            latent2image(
                                model=self.vae, latents=inverted_x[1].to(self.vae.dtype)
                            )[0]
                        )
                    ),
                    np.uint8(
                        255 * np.array(edited_image[0].permute(1, 2, 0).cpu().detach())
                    ),
                ),
                1,
            )
        )
