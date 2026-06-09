import h5py, os, sys

ds_dir = "/home/minum/26CS/MoNaVLA/ROS_action/mobile_vla_dataset_v5"
files = sorted([f for f in os.listdir(ds_dir) if f.endswith('.h5')])
print(f"총 에피소드 수: {len(files)}")

ep = files[0]
print(f"\n첫 번째 파일: {ep}")
with h5py.File(os.path.join(ds_dir, ep), 'r') as f:
    def pk(name, obj):
        print(f"  {name}  [{type(obj).__name__}]  shape={getattr(obj,'shape','')}")
    f.visititems(pk)
