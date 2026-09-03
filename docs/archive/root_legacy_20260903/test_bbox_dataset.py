import torch
import torch.utils.data as data
from robovlm_nav.datasets.nav_h5_dataset_impl import MobileVLAH5Dataset
from torch.utils.data import DataLoader

ds = MobileVLAH5Dataset(
    data_dir="/home/billy/25-1kp/MoNaVLA/ROS_action/mobile_vla_dataset_v5",
    episode_pattern="episode_*.h5",
    window_size=1,
    fwd_pred_next_n=1,
    discrete_action=False,
    use_bbox_target=True,
    grounding_prefix=True
)

for i in range(5):
    ret = ds[i]
    print(f"Sample {i}:")
    print(ret['text'])

