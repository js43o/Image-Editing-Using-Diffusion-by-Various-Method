import torch
from compel import Compel, ReturnedEmbeddingsType

from models.p2p.attention_control import register_attention_control
from utils.utils import init_latent


def p2p_guidance_diffusion_step(
    model, controller, latents, context, t, guidance_scale, low_resource=False
):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])[
            "sample"
        ]
        noise_prediction_text = model.unet(
            latents, t, encoder_hidden_states=context[1]
        )["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)[
            "sample"
        ]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_prediction_text - noise_pred_uncond
    )
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents


@torch.no_grad()
def p2p_guidance_forward(
    model,
    prompt,
    controller,
    num_inference_steps: int = 50,
    guidance_scale=7.5,
    generator=None,
    latent=None,
    uncond_embeddings=None,
):
    batch_size = len(prompt)
    register_attention_control(model, controller)
    height = width = 512

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings_ = model.text_encoder(
            uncond_input.input_ids.to(model.device)
        )[0]
    else:
        uncond_embeddings_ = None

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(model.scheduler.timesteps):
        if uncond_embeddings_ is None:
            context = torch.cat(
                [uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings]
            )
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = p2p_guidance_diffusion_step(
            model, controller, latents, context, t, guidance_scale, low_resource=False
        )

    return latents, latent


@torch.no_grad()
def p2p_guidance_forward_single_branch(
    model,
    prompt,
    controller,
    num_inference_steps: int = 50,
    guidance_scale=7.5,
    generator=None,
    latent=None,
    uncond_embeddings=None,
):
    batch_size = len(prompt)
    register_attention_control(model, controller)
    height = width = 512

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]

    uncond_input = model.tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(model.scheduler.timesteps):
        context = torch.cat(
            [torch.cat([uncond_embeddings[i], uncond_embeddings_[1:]]), text_embeddings]
        )
        latents = p2p_guidance_diffusion_step(
            model, controller, latents, context, t, guidance_scale, low_resource=False
        )

    return latents, latent


def direct_inversion_p2p_guidance_diffusion_step(
    model,
    controller,
    latents,
    context,
    context_p,
    t,
    add_time_ids,
    guidance_scale,
    noise_loss,
    add_offset=True,
):
    print("✅ direct_inversion_p2p_guidance_diffusion_step()")

    added_cond_kwargs = {"text_embeds": context_p, "time_ids": add_time_ids}
    latents_input = torch.cat([latents] * 2)

    noise_pred = model.unet(
        latents_input,
        t,
        encoder_hidden_states=context,
        added_cond_kwargs=added_cond_kwargs,
    )["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)

    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_prediction_text - noise_pred_uncond
    )
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    if add_offset:
        latents = torch.concat((latents[:1] + noise_loss[:1], latents[1:]))
    latents = controller.step_callback(latents)
    return latents


def direct_inversion_p2p_guidance_diffusion_step_add_target(
    model,
    controller,
    latents,
    context,
    t,
    guidance_scale,
    noise_loss,
    low_resource=False,
    add_offset=True,
):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])[
            "sample"
        ]
        noise_prediction_text = model.unet(
            latents, t, encoder_hidden_states=context[1]
        )["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)[
            "sample"
        ]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_prediction_text - noise_pred_uncond
    )
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    if add_offset:
        latents = torch.concat(
            (latents[:1] + noise_loss[:1], latents[1:] + noise_loss[1:])
        )
    latents = controller.step_callback(latents)
    return latents


@torch.no_grad()
def direct_inversion_p2p_guidance_forward(
    model,
    prompt,
    controller,
    latent=None,
    num_inference_steps: int = 50,
    guidance_scale=7.5,
    generator=None,
    noise_loss_list=None,
    add_offset=True,
):
    print("✅ direct_inversion_p2p_guidance_forward()")

    register_attention_control(model, controller)

    compel = Compel(
        tokenizer=[model.tokenizer, model.tokenizer_2],
        text_encoder=[model.text_encoder, model.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True],
    )

    prompt_embeds, pooled_prompt_embeds = compel(prompt)
    negative_prompt_embeds, negative_pooled_prompt_embeds = compel([""] * len(prompt))

    context = torch.cat([negative_prompt_embeds, prompt_embeds])
    context_p = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])

    # 512 대신 사용
    model.vae_scale_factor = 2 ** (len(model.vae.config.block_out_channels) - 1)
    model.default_sample_size = model.unet.config.sample_size

    height = model.default_sample_size * model.vae_scale_factor
    width = model.default_sample_size * model.vae_scale_factor

    original_size = (height, width)
    target_size = (height, width)
    crops_coords_top_left = (0, 0)
    add_time_ids = list(original_size + crops_coords_top_left + target_size)

    passed_add_embed_dim = (
        model.unet.config.addition_time_embed_dim * len(add_time_ids)
        + model.text_encoder_2.config.projection_dim
    )
    expected_add_embed_dim = model.unet.add_embedding.linear_1.in_features

    if expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )

    add_time_ids = torch.tensor([add_time_ids], dtype=model.unet.dtype).to(model.device)
    batch_size = prompt_embeds.shape[0]
    add_time_ids = add_time_ids.repeat(batch_size, 1)
    add_time_ids = torch.cat([add_time_ids, add_time_ids])

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)

    for i, t in enumerate(model.scheduler.timesteps):
        latents = direct_inversion_p2p_guidance_diffusion_step(
            model,
            controller,
            latents,
            context,
            context_p,
            t,
            add_time_ids,
            guidance_scale,
            noise_loss_list[i],
            add_offset=add_offset,
        )

    return latents, latent


@torch.no_grad()
def direct_inversion_p2p_guidance_forward_add_target(
    model,
    prompt,
    controller,
    latent=None,
    num_inference_steps: int = 50,
    guidance_scale=7.5,
    generator=None,
    noise_loss_list=None,
    add_offset=True,
):
    print("🚫 direct_inversion_p2p_guidance_forward_add_target()")
    batch_size = len(prompt)
    register_attention_control(model, controller)
    height = width = 512

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]

    uncond_input = model.tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(model.scheduler.timesteps):

        context = torch.cat([uncond_embeddings, text_embeddings])
        latents = direct_inversion_p2p_guidance_diffusion_step_add_target(
            model,
            controller,
            latents,
            context,
            t,
            guidance_scale,
            noise_loss_list[i],
            low_resource=False,
            add_offset=add_offset,
        )

    return latents, latent
