# available editing methods combination: 
    # [ddim, null-text-inversion, negative-prompt-inversion, directinversion] + 
    # [p2p, masactrl, pix2pix_zero, pnp]

# p2p only!
conda run -n p2p python -u main.py --data_path img \
                --output_path output \
                --edit_method_list directinversion+p2p

# masactrl only!
conda run -n masactrl python -u main.py --data_path img \
                --output_path output \
                --edit_method_list null-text-inversion+masactrl
# pix2pix_zero only!
conda run -n pix2pix_zero python -u main.py --data_path img \
                --output_path output \
                --edit_method_list ddim+pix2pix_zero

# pnp only!
conda run -n pnp python -u main.py --data_path img \
                --output_path output \
                --edit_method_list negative-prompt-inversion+pnp