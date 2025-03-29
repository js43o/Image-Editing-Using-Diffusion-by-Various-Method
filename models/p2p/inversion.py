import torch
import torch.nn.functional as nnf
from torch.optim.adam import Adam
from compel import Compel, ReturnedEmbeddingsType

from models.p2p.attention_control import register_attention_control
from utils.utils import image2latent, latent2image


class DirectInversion:
    def prev_step(self, model_output, timestep: int, sample):
        print("✅ prev_step()")
        prev_timestep = (
            timestep
            - self.scheduler.config.num_train_timesteps
            // self.scheduler.num_inference_steps
        )
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.scheduler.final_alpha_cumprod
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

    def next_step(self, model_output, timestep: int, sample):
        print("✅ next_step()")
        timestep, next_timestep = (
            min(
                timestep
                - self.scheduler.config.num_train_timesteps
                // self.scheduler.num_inference_steps,
                999,
            ),
            timestep,
        )
        alpha_prod_t = (
            self.scheduler.alphas_cumprod[timestep]
            if timestep >= 0
            else self.scheduler.final_alpha_cumprod
        )
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (
            sample - beta_prod_t**0.5 * model_output
        ) / alpha_prod_t**0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = (
            alpha_prod_t_next**0.5 * next_original_sample + next_sample_direction
        )
        return next_sample

    def get_noise_pred_single(self, latents, t, context, context_p, add_time_ids):
        print("✅ get_noise_pred_single()")

        latents = self.scheduler.scale_model_input(latents, t)
        added_cond_kwargs = {"text_embeds": context_p, "time_ids": add_time_ids}
        noise_pred = self.model.unet(
            latents,
            t,
            encoder_hidden_states=context,
            added_cond_kwargs=added_cond_kwargs,
        )["sample"]
        return noise_pred

    def get_noise_pred(
        self, latents, t, guidance_scale, is_forward=True, context=None, context_p=None
    ):
        print("🚫 get_noise_pred()")
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else guidance_scale
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)[
            "sample"
        ]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_prediction_text - noise_pred_uncond
        )
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        print("✅ init_prompt()", prompt)
        """
        Null-text inversion과 달리, 소스 프롬프트와 편집 프롬프트가 한꺼번에 리스트 형태로 주어짐 (Null-text inversion에서는 단일 문자열)
        ➡️ 자연스럽게 임베딩의 배치 크기도 1이 아닌 2가 됨
        ➡️ add_time_ids의 0번째 차원 수가 2가 됨
        """
        compel = Compel(
            tokenizer=[self.model.tokenizer, self.model.tokenizer_2],
            text_encoder=[self.model.text_encoder, self.model.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
        )

        prompt_embeds, pooled_prompt_embeds = compel(prompt)
        negative_prompt_embeds, negative_pooled_prompt_embeds = compel(
            [""] * len(prompt)
        )

        self.model.vae_scale_factor = 2 ** (
            len(self.model.vae.config.block_out_channels) - 1
        )
        self.model.default_sample_size = self.model.unet.config.sample_size

        height = self.model.default_sample_size * self.model.vae_scale_factor
        width = self.model.default_sample_size * self.model.vae_scale_factor

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
            self.model.device
        )
        batch_size = prompt_embeds.shape[0]
        add_time_ids = add_time_ids.repeat(batch_size, 1)

        self.context = torch.cat([negative_prompt_embeds, prompt_embeds])
        self.context_p = torch.cat(
            [negative_pooled_prompt_embeds, pooled_prompt_embeds]
        )

        self.add_time_ids = torch.cat([add_time_ids, add_time_ids])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        print("✅ ddim_loop()")
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_p, cond_embeddings_p = self.context_p.chunk(2)
        add_time_ids1, add_time_ids2 = self.add_time_ids.chunk(2)

        all_latent = [latent]
        latent = latent.clone().detach()

        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[
                len(self.model.scheduler.timesteps) - i - 1
            ]
            noise_pred = self.get_noise_pred_single(
                latent,
                t,
                cond_embeddings[[0]],
                cond_embeddings_p[[0]],
                add_time_ids2[[0]],
            )
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)

        return all_latent

    @torch.no_grad()
    def ddim_null_loop(self, latent):
        print("🚫 ddim_null_loop()")
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings = uncond_embeddings[[0]]
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[
                len(self.model.scheduler.timesteps) - i - 1
            ]
            noise_pred = self.get_noise_pred_single(latent, t, uncond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @torch.no_grad()
    def ddim_with_guidance_scale_loop(self, latent, guidance_scale):
        print("🚫 ddim_with_guidance_scale_loop()")
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings = uncond_embeddings[[0]]
        cond_embeddings = cond_embeddings[[0]]
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[
                len(self.model.scheduler.timesteps) - i - 1
            ]
            uncond_noise_pred = self.get_noise_pred_single(latent, t, uncond_embeddings)
            cond_noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            noise_pred = uncond_noise_pred + guidance_scale * (
                cond_noise_pred - uncond_noise_pred
            )
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        print("✅ ddim_inversion()")
        latent = image2latent(self.model.vae, image)
        image_rec = latent2image(self.model.vae, latent)[0]
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    @torch.no_grad()
    def ddim_null_inversion(self, image):
        print("✅ ddim_null_inversion()")
        latent = image2latent(self.model.vae, image)
        image_rec = latent2image(self.model.vae, latent)[0]
        ddim_latents = self.ddim_null_loop(latent)
        return image_rec, ddim_latents

    @torch.no_grad()
    def ddim_with_guidance_scale_inversion(self, image, guidance_scale):
        print("🚫 ddim_with_guidance_scale_inversion()")
        latent = image2latent(self.model.vae, image)
        image_rec = latent2image(self.model.vae, latent)[0]
        ddim_latents = self.ddim_with_guidance_scale_loop(latent, guidance_scale)
        return image_rec, ddim_latents

    # Null-text inversion의 null_optimization()과 유사한 역할 (손실 계산 후 반환)
    def offset_calculate(self, latents, num_inner_steps, epsilon, guidance_scale):
        print("✅ offset_calculate()")

        noise_loss_list = []
        latent_cur = torch.concat([latents[-1]] * (self.context.shape[0] // 2))
        for i in range(self.num_ddim_steps):
            latent_prev = torch.concat(
                [latents[len(latents) - i - 2]] * latent_cur.shape[0]
            )
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred = self.get_noise_pred_single(
                    torch.concat([latent_cur] * 2),
                    t,
                    self.context,
                    self.context_p,
                    self.add_time_ids,
                )
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred_w_guidance = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
                latents_prev_rec, _ = self.prev_step(
                    noise_pred_w_guidance, t, latent_cur
                )
                loss = latent_prev - latents_prev_rec

            noise_loss_list.append(loss.detach())
            latent_cur = latents_prev_rec + loss

        return noise_loss_list

    def invert(
        self,
        image_gt,
        prompt,
        guidance_scale,
        num_inner_steps=10,
        early_stop_epsilon=1e-5,
    ):
        print("✅ invert()")
        self.init_prompt(prompt)
        register_attention_control(self.model, None)

        image_rec, ddim_latents = self.ddim_inversion(image_gt)

        noise_loss_list = self.offset_calculate(
            ddim_latents, num_inner_steps, early_stop_epsilon, guidance_scale
        )
        return image_gt, image_rec, ddim_latents, noise_loss_list

    def invert_without_attn_controller(
        self,
        image_gt,
        prompt,
        guidance_scale,
        num_inner_steps=10,
        early_stop_epsilon=1e-5,
    ):
        print("🚫 invert_without_attn_controller()")
        self.init_prompt(prompt)

        image_rec, ddim_latents = self.ddim_inversion(image_gt)

        noise_loss_list = self.offset_calculate(
            ddim_latents, num_inner_steps, early_stop_epsilon, guidance_scale
        )
        return image_gt, image_rec, ddim_latents, noise_loss_list

    def invert_with_guidance_scale_vary_guidance(
        self,
        image_gt,
        prompt,
        inverse_guidance_scale,
        forward_guidance_scale,
        num_inner_steps=10,
        early_stop_epsilon=1e-5,
    ):
        print("🚫 invert_with_guidance_scale_vary_guidance()")
        self.init_prompt(prompt)
        register_attention_control(self.model, None)

        image_rec, ddim_latents = self.ddim_with_guidance_scale_inversion(
            image_gt, inverse_guidance_scale
        )

        noise_loss_list = self.offset_calculate(
            ddim_latents, num_inner_steps, early_stop_epsilon, forward_guidance_scale
        )
        return image_gt, image_rec, ddim_latents, noise_loss_list

    def null_latent_calculate(self, latents, num_inner_steps, epsilon, guidance_scale):
        print("🚫 null_latent_calculate()")
        noise_loss_list = []
        latent_cur = torch.concat([latents[-1]] * (self.context.shape[0] // 2))
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        for i in range(self.num_ddim_steps):
            latent_prev = torch.concat(
                [latents[len(latents) - i - 2]] * latent_cur.shape[0]
            )
            t = self.model.scheduler.timesteps[i]

            if num_inner_steps != 0:
                uncond_embeddings = uncond_embeddings.clone().detach()
                uncond_embeddings.requires_grad = True
                optimizer = Adam([uncond_embeddings], lr=1e-2 * (1.0 - i / 100.0))
                for j in range(num_inner_steps):
                    latents_input = torch.cat([latent_cur] * 2)
                    noise_pred = self.model.unet(
                        latents_input,
                        t,
                        encoder_hidden_states=torch.cat(
                            [uncond_embeddings, cond_embeddings]
                        ),
                    )["sample"]
                    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_prediction_text - noise_pred_uncond
                    )

                    latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)[0]

                    loss = nnf.mse_loss(latents_prev_rec[[0]], latent_prev[[0]])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_item = loss.item()

                    if loss_item < epsilon + i * 2e-5:
                        break

            with torch.no_grad():
                noise_pred = self.get_noise_pred_single(
                    torch.concat([latent_cur] * 2), t, self.context
                )
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred_w_guidance = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
                latents_prev_rec, _ = self.prev_step(
                    noise_pred_w_guidance, t, latent_cur
                )

                latent_cur = self.get_noise_pred(
                    latent_cur,
                    t,
                    guidance_scale,
                    False,
                    torch.cat([uncond_embeddings, cond_embeddings]),
                )[0]
                loss = latent_cur - latents_prev_rec

            noise_loss_list.append(loss.detach())
            latent_cur = latents_prev_rec + loss

        return noise_loss_list

    def invert_null_latent(
        self,
        image_gt,
        prompt,
        guidance_scale,
        num_inner_steps=10,
        early_stop_epsilon=1e-5,
    ):
        print("🚫 invert_null_latent()")
        self.init_prompt(prompt)
        register_attention_control(self.model, None)

        image_rec, ddim_latents = self.ddim_inversion(image_gt)

        latent_list = self.null_latent_calculate(
            ddim_latents, num_inner_steps, early_stop_epsilon, guidance_scale
        )
        return image_gt, image_rec, ddim_latents, latent_list

    def offset_calculate_not_full(
        self, latents, num_inner_steps, epsilon, guidance_scale, scale
    ):
        print("🚫 offset_calculate_not_full()")
        noise_loss_list = []
        latent_cur = torch.concat([latents[-1]] * (self.context.shape[0] // 2))
        for i in range(self.num_ddim_steps):
            latent_prev = torch.concat(
                [latents[len(latents) - i - 2]] * latent_cur.shape[0]
            )
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred = self.get_noise_pred_single(
                    torch.concat([latent_cur] * 2), t, self.context
                )
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred_w_guidance = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
                latents_prev_rec, _ = self.prev_step(
                    noise_pred_w_guidance, t, latent_cur
                )
                loss = latent_prev - latents_prev_rec
                loss = loss * scale

            noise_loss_list.append(loss.detach())
            latent_cur = latents_prev_rec + loss

        return noise_loss_list

    def invert_not_full(
        self,
        image_gt,
        prompt,
        guidance_scale,
        num_inner_steps=10,
        early_stop_epsilon=1e-5,
        scale=1.0,
    ):
        print("🚫 invert_not_full()")
        self.init_prompt(prompt)
        register_attention_control(self.model, None)

        image_rec, ddim_latents = self.ddim_inversion(image_gt)

        noise_loss_list = self.offset_calculate_not_full(
            ddim_latents, num_inner_steps, early_stop_epsilon, guidance_scale, scale
        )
        return image_gt, image_rec, ddim_latents, noise_loss_list

    def offset_calculate_skip_step(
        self, latents, num_inner_steps, epsilon, guidance_scale, skip_step
    ):
        print("🚫 offset_calculate_skip_step()")
        noise_loss_list = []
        latent_cur = torch.concat([latents[-1]] * (self.context.shape[0] // 2))
        for i in range(self.num_ddim_steps):
            latent_prev = torch.concat(
                [latents[len(latents) - i - 2]] * latent_cur.shape[0]
            )
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred = self.get_noise_pred_single(
                    torch.concat([latent_cur] * 2), t, self.context
                )
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred_w_guidance = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
                latents_prev_rec, _ = self.prev_step(
                    noise_pred_w_guidance, t, latent_cur
                )
                if (i % skip_step) == 0:
                    loss = latent_prev - latents_prev_rec
                else:
                    loss = torch.zeros_like(latent_prev)

            noise_loss_list.append(loss.detach())
            latent_cur = latents_prev_rec + loss

        return noise_loss_list

    def invert_skip_step(
        self,
        image_gt,
        prompt,
        guidance_scale,
        skip_step,
        num_inner_steps=10,
        early_stop_epsilon=1e-5,
        scale=1.0,
    ):
        print("🚫 invert_skip_step()")
        self.init_prompt(prompt)
        register_attention_control(self.model, None)

        image_rec, ddim_latents = self.ddim_inversion(image_gt)

        noise_loss_list = self.offset_calculate_skip_step(
            ddim_latents, num_inner_steps, early_stop_epsilon, guidance_scale, skip_step
        )
        return image_gt, image_rec, ddim_latents, noise_loss_list

    def __init__(self, model, num_ddim_steps):
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.prompt = None
        self.context = None
        self.num_ddim_steps = num_ddim_steps
