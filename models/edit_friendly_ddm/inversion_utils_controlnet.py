import torch
import os
from .inversion_utils import encode_text, reverse_step

def inversion_reverse_process_controlnet(model,
                    controlnet,
                    xT, 
                    etas = 0,
                    prompts = "",
                    cfg_scales = None,
                    prog_bar = False,
                    zs = None,
                    controller=None,
                    asyrp = False,
                    control_image=None,
                    control_gamma=0.5,
                    control_scale=0.8,
                    control_guidance_end=0.8):

    batch_size = len(prompts)

    cfg_scales_tensor = torch.Tensor(cfg_scales).view(-1,1,1,1).to(model.device)

    text_embeddings = encode_text(model, prompts)
    uncond_embedding = encode_text(model, [""] * batch_size)

    if etas is None: etas = 0
    if type(etas) in [int, float]: etas = [etas]*model.scheduler.num_inference_steps
    assert len(etas) == model.scheduler.num_inference_steps
    timesteps = model.scheduler.timesteps.to(model.device)

    xt = xT.expand(batch_size, -1, -1, -1)
    op = timesteps[-zs.shape[0]:] 

    t_to_idx = {int(v):k for k,v in enumerate(timesteps[-zs.shape[0]:])}

    for t in op:
        idx = model.scheduler.num_inference_steps-t_to_idx[int(t)]-(model.scheduler.num_inference_steps-zs.shape[0]+1)
        
        # Calculate current progress (0.0 to 1.0)
        # t goes from T to 0. So progress is (T-t)/T
        # But here t is a tensor or int representing the timestep value (e.g. 981, 961...)
        # model.scheduler.config.num_train_timesteps is usually 1000.
        current_step_ratio = 1 - (t / model.scheduler.config.num_train_timesteps)
        
        # ControlNet forward pass
        # apply ControlNet to both unconditional and conditional branches
        # assume control_image is already properly preprocessed and on device
        
        # Check if we should apply controlnet based on guidance end
        if current_step_ratio < control_guidance_end:
            # Unconditional ControlNet
            down_block_res_samples_uncond, mid_block_res_sample_uncond = controlnet(
                xt,
                t,
                encoder_hidden_states=uncond_embedding,
                controlnet_cond=control_image,
                return_dict=False,
            )

            # Conditional ControlNet
            down_block_res_samples_cond, mid_block_res_sample_cond = controlnet(
                xt,
                t,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=control_image,
                return_dict=False,
            )
            
            # Apply control scale
            down_block_res_samples_uncond = [sample * control_scale for sample in down_block_res_samples_uncond]
            mid_block_res_sample_uncond = mid_block_res_sample_uncond * control_scale
            down_block_res_samples_cond = [sample * control_scale for sample in down_block_res_samples_cond]
            mid_block_res_sample_cond = mid_block_res_sample_cond * control_scale
            
        else:
            down_block_res_samples_uncond = None
            mid_block_res_sample_uncond = None
            down_block_res_samples_cond = None
            mid_block_res_sample_cond = None


        ## Unconditional embedding
        with torch.no_grad():
            uncond_out = model.unet.forward(xt, timestep =  t, 
                                            encoder_hidden_states = uncond_embedding,
                                            down_block_additional_residuals=down_block_res_samples_uncond,
                                            mid_block_additional_residual=mid_block_res_sample_uncond)

            ## Conditional embedding  
        if prompts:  
            with torch.no_grad():
                cond_out = model.unet.forward(xt, timestep =  t, 
                                                encoder_hidden_states = text_embeddings,
                                                down_block_additional_residuals=down_block_res_samples_cond,
                                                mid_block_additional_residual=mid_block_res_sample_cond)
            
        
        z = zs[idx] if not zs is None else None
        z = z.expand(batch_size, -1, -1, -1)
        
        # Relaxed Inversion: Blend inverted noise with random noise
        if z is not None:
            z_rand = torch.randn_like(z)
            
            # Use Analytical Preserving Blending (SLERP-like linear approximation)
            # This guarantees that if input variances are 1.0, output variance is 1.0
            denom = (control_gamma**2 + (1-control_gamma)**2) ** 0.5
            z = ((control_gamma * z) + ((1 - control_gamma) * z_rand)) / denom

        if prompts:
            ## classifier free guidance
            noise_pred = uncond_out.sample + cfg_scales_tensor * (cond_out.sample - uncond_out.sample)
        else: 
            noise_pred = uncond_out.sample
        # 2. compute less noisy image and set x_t -> x_t-1  
        xt = reverse_step(model, noise_pred, t, xt, eta = etas[idx], variance_noise = z) 
        if controller is not None:
            xt = controller.step_callback(xt)        
    return xt, zs
