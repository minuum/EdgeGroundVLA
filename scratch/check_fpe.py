import json

with open('/home/minum/26CS/MoNaVLA/docs/v5/closed_loop_eval/rollout_metrics.json') as f:
    d = json.load(f)
pp = d['per_path']

paths = ['center_straight','center_left','center_right',
         'left_straight','left_left','left_right',
         'right_straight','right_left','right_right']

for model in ['exp11','step2','exp49']:
    print(f'=== {model} ===')
    for p in paths:
        eps = pp[model].get(p, [])
        if eps:
            fpe = eps[0]['fpe']
            tld = eps[0]['tld']
            success = eps[0]['success']
            ep_id = eps[0].get('episode','')
            print(f'  {p}: FPE={fpe:.2f}m TLD={tld:.2f}m ok={success}')
        else:
            print(f'  {p}: NO DATA')
