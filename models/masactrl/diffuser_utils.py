"""
Util functions based on Diffuser framework.
"""

import torch
import numpy as np

from tqdm import tqdm
from PIL import Image

from diffusers import StableDiffusionXLPipeline
from compel import Compel, ReturnedEmbeddingsType


class MasaCtrlPipeline(StableDiffusionXLPipeline):

    def next_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta=0.0,
        verbose=False,
    ):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(
            timestep
            - self.scheduler.config.num_train_timesteps
            // self.scheduler.num_inference_steps,
            999,
        )
        alpha_prod_t = (
            self.scheduler.alphas_cumprod[timestep]
            if timestep >= 0
            else self.scheduler.final_alpha_cumprod
        )
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_next) ** 0.5 * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta: float = 0.0,
        verbose=False,
    ):
        """
        predict the sampe the next step in the denoise process.
        """
        prev_timestep = (
            timestep
            - self.scheduler.config.num_train_timesteps
            // self.scheduler.num_inference_steps
        )
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep > 0
            else self.scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0

    @torch.no_grad()
    def image2latent(self, image):
        DEVICE = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        if type(image) is Image:
            image = np.array(image)
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        # input image density range [-1, 1]
        latents = self.vae.encode(image)["latent_dist"].mean
        latents = latents * 0.13025
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type="np"):
        latents = 1 / 0.13025 * latents.detach()
        image = self.vae.decode(latents)["sample"]
        if return_type == "np":
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)

        return image

    def latent2image_grad(self, latents):
        latents = 1 / 0.13025 * latents
        image = self.vae.decode(latents)["sample"]

        return image  # range [-1, 1]

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        batch_size=1,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        latents=None,
        unconditioning=None,
        neg_prompt=None,
        ref_intermediate_latents=None,
        return_intermediates=False,
        noise_loss_list=None,
        **kwds,
    ):
        DEVICE = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        if isinstance(prompt, list):
            batch_size = len(prompt)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # prompt embedding
        compel = Compel(
            tokenizer=[self.tokenizer, self.tokenizer_2],
            text_encoder=[self.text_encoder, self.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
        )

        prompt_embeds, pooled_prompt_embeds = compel(prompt)
        negative_prompt_embeds, negative_pooled_prompt_embeds = compel(
            [""] * len(prompt)
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.default_sample_size = self.unet.config.sample_size

        height = self.default_sample_size * self.vae_scale_factor
        width = self.default_sample_size * self.vae_scale_factor

        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids)
            + self.text_encoder_2.config.projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=self.unet.dtype).to(DEVICE)
        batch_size = prompt_embeds.shape[0]
        add_time_ids = add_time_ids.repeat(batch_size, 1)
        add_time_ids = torch.cat([add_time_ids, add_time_ids])

        context = torch.cat([negative_prompt_embeds, prompt_embeds])
        context_p = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])

        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)

        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):
            model_inputs = torch.cat([latents] * 2)

            # predict the noise
            added_cond_kwargs = {
                "text_embeds": context_p,
                "time_ids": add_time_ids,
            }
            noise_pred = self.unet(
                model_inputs,
                t,
                encoder_hidden_states=context,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

            noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncon + guidance_scale * (
                noise_pred_con - noise_pred_uncon
            )

            # compute the previous noise sample x_t -> x_t-1
            latents, pred_x0 = self.step(noise_pred, t, latents)
            if noise_loss_list is not None:
                latents = torch.concat(
                    (latents[:1] + noise_loss_list[i][:1], latents[1:])
                )

            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        image = self.latent2image(latents, return_type="pt")
        if return_intermediates:
            pred_x0_list = [
                self.latent2image(img, return_type="pt") for img in pred_x0_list
            ]
            latents_list = [
                self.latent2image(img, return_type="pt") for img in latents_list
            ]
            return image, pred_x0_list, latents_list

        return image

    @torch.no_grad()
    def invert(
        self,
        image: torch.Tensor,
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        return_intermediates=False,
        **kwds,
    ):
        """
        invert a real image into noise map with deterministic DDIM inversion
        """
        print("✅ invert()", prompt)
        batch_size = image.shape[0]
        print("📦 batch_size =", batch_size)

        DEVICE = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        if isinstance(prompt, list):
            if batch_size == 1:
                image = image.expand(len(prompt), -1, -1, -1)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # prompt embedding
        compel = Compel(
            tokenizer=[self.tokenizer, self.tokenizer_2],
            text_encoder=[self.text_encoder, self.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
        )

        prompt_embeds, pooled_prompt_embeds = compel(prompt)
        negative_prompt_embeds, negative_pooled_prompt_embeds = compel("")

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.default_sample_size = self.unet.config.sample_size

        height = self.default_sample_size * self.vae_scale_factor
        width = self.default_sample_size * self.vae_scale_factor

        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids)
            + self.text_encoder_2.config.projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=self.unet.dtype).to(DEVICE)
        batch_size = prompt_embeds.shape[0]
        add_time_ids = add_time_ids.repeat(batch_size, 1)
        add_time_ids = torch.cat([add_time_ids, add_time_ids])

        context = torch.cat([negative_prompt_embeds, prompt_embeds])
        context_p = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])

        # define initial latents
        latents = self.image2latent(image)
        start_latents = latents

        print("📦 latents =", latents.shape)

        # interative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        print("Valid timesteps: ", reversed(self.scheduler.timesteps))

        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(
            tqdm(reversed(self.scheduler.timesteps), desc="DDIM Inversion")
        ):
            model_inputs = torch.cat([latents] * 2)

            # predict the noise
            added_cond_kwargs = {
                "text_embeds": context_p,
                "time_ids": add_time_ids,
            }
            noise_pred = self.unet(
                model_inputs,
                t,
                encoder_hidden_states=context,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

            noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncon + guidance_scale * (
                noise_pred_con - noise_pred_uncon
            )

            # compute the previous noise sample x_t-1 -> x_t
            latents, pred_x0 = self.next_step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        if return_intermediates:
            # return the intermediate laters during inversion
            # pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            return latents, latents_list

        return latents, start_latents
