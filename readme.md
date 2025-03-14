# Image Editing Using Diffusion by Various Method
This project is useful for the person who want to edit images using diffusion model by "Null-text-inversion, Negative-prompt-inversion, Directinversion" combined with "Prompt-to-Prompt, Masactrl, Pix2Pix Zero, PnP".    

## To-Do
- [x] Complete Prompt-to-Prompt
- [x] Complete MasaCtrl
- [ ] Complete Pix2Pix Zero
- [x] Complete PnP  

## 🌱 Environment Setting
**For Prompt-to-Prompt**
```bash
conda env create --file env/p2p.yaml
```
**For MasaCtrl**
```bash
conda env create --file env/masactrl.yaml
```
**For Pix2Pix Zero**
```bash
conda env create --file env/pix2pix_zero.yaml
```
**For PnP**
```bash
conda env create --file env/pnp.yaml
```

## 🚀 Run
### 📝 How to write prompt
In **main.py**
```python
original_prompt = "A white horse running in the field"
editing_prompt = "Water color of a white horse running in the field"
image_path = "./img/horse.png"
editing_instruction = "" #You can write the instruction on it
blended_word = [] #Ex. ["horse", "dog"] if you want to change word "horse" in source prompt to word "dog" in target prompt
```
### 🎯 How to run to get result  
You can obtain images by combining below:    

| 🔒 Preserving source Image | 🎨 Editing image by target prompt|
| :- | :- |
| null-text-inversion | p2p |
| negative-prompt-inversion | masactrl |
| directinversion | pix2pix_zero |
| ddim | pnp |    

(You can observe the results obtained without the image preservation technique by using "ddim".)

Since the Conda environment depends on the specific editing method, please make sure to execute commands using:

Please refer to **run.sh**  
```bash
conda run -n p2p python -u main.py --data_path img \
                --output_path output \
                --edit_method_list directinversion+p2p
```
And Finally, You can find the result in **output** directory.

## Acknowledgement
This code has been modified based on the [PnP_Inversion](https://github.com/cure-lab/PnPInversion/tree/main).    
Following the implementation from [null-text inversion](https://github.com/google/prompt-to-prompt/#null-text-inversion-for-editing-real-images), [negative-prompt inversion](https://arxiv.org/abs/2305.16807), [Direct inversion](https://arxiv.org/abs/2310.01506), [prompt-to-prompt](https://github.com/google/prompt-to-prompt), [MasaCtrl](https://github.com/TencentARC/MasaCtrl), [pix2pix-zero](https://github.com/pix2pixzero/pix2pix-zero) , [Plug-and-Play](https://github.com/MichalGeyer/plug-and-play).     
Sincerely thank all contributors.