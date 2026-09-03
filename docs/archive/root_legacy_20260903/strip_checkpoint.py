import torch
import gc

ckpt_path = "epoch_epoch=05-val_loss=val_loss=0.044.ckpt"
out_path = "v3_exp07_lora_light.ckpt"

print(f"Loading {ckpt_path} onto CPU...")
# map_location='cpu' avoids GPU OOM, but we only have 10GB RAM. 
# 8.7GB is quite large, but let's try.
checkpoint = torch.load(ckpt_path, map_location="cpu")

keys_to_delete = ["optimizer_states", "lr_schedulers", "callbacks"]
for k in keys_to_delete:
    if k in checkpoint:
        print(f"Removing {k}...")
        del checkpoint[k]

print(f"Saving to {out_path}...")
torch.save(checkpoint, out_path)
print("Done!")
