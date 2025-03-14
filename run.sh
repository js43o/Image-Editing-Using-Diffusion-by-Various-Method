# available editing methods combination: 
    # [ddim, null-text-inversion, negative-prompt-inversion, directinversion, inversion-free-editing] + 
    # [p2p, masactrl, pix2pix_zero, pnp]

# p2p only!
# conda init
# conda activate stable_diff
# python -u main.py --data_path img \
#                 --output_path output \
#                 --edit_method_list directinversion+p2p

# masactrl only!
# conda activate masactrl
# python -u main.py --data_path img \
#                 --output_path output \
#                 --edit_method_list null-text-inversion+masactrl
# pix2pix_zero only!
# conda init
# conda activate pix2pix_zero
# python -u main.py --data_path img \
#                 --output_path output \
#                 --edit_method_list ddim+pix2pix_zero

# pnp only!
conda init
conda activate pnp
python -u main.py --data_path img \
                --output_path output \
                --edit_method_list negative-prompt-inversion+pnp