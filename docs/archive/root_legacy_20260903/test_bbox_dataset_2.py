import torch
from robovlm_nav.datasets.nav_h5_dataset_impl import MobileVLAH5Dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/kosmos-2-patch14-224")

ds = MobileVLAH5Dataset(
    data_dir="/home/billy/25-1kp/MoNaVLA/ROS_action/mobile_vla_dataset_v5",
    episode_pattern="episode_*.h5",
    window_size=1,
    fwd_pred_next_n=1,
    discrete_action=False,
    use_bbox_target=True,
    grounding_prefix=True,
    tokenizer=tokenizer
)

for i in range(2):
    ret = ds[i]
    print(f"Sample {i} Raw Text:")
    print(ret['raw_text'])
    text_ids = ret['text']
    text_ids = text_ids[text_ids != 0]
    decoded = tokenizer.decode(text_ids)
    print(f"Sample {i} Decoded:")
    print(decoded)
