import json, sys

with open('/home/minum/26CS/MoNaVLA/docs/v5/closed_loop_eval/exp47_closed_loop_results.json') as f:
    d = json.load(f)
eps = d['episodes']
print('episodes type:', type(eps))
if isinstance(eps, dict):
    print('episodes keys (first 5):', list(eps.keys())[:5])
    k0 = list(eps.keys())[0]
    ep0 = eps[k0]
elif isinstance(eps, list):
    print('episodes len:', len(eps))
    ep0 = eps[0]

print('episode[0] keys:', list(ep0.keys()))
for k in ep0:
    v = ep0[k]
    if isinstance(v, list) and len(v) > 0:
        print(f'  {k} (list len={len(v)}): first={str(v[0])[:100]}')
    else:
        print(f'  {k}: {str(v)[:150]}')

print()
# rollout_metrics.json 확인
with open('/home/minum/26CS/MoNaVLA/docs/v5/closed_loop_eval/rollout_metrics.json') as f:
    r = json.load(f)
pp = r['per_path']
print('rollout_metrics per_path models:', list(pp.keys()))
print('paths:', list(pp['step2'].keys()))
ep = pp['step2']['center_straight'][0]
print('\nstep2 center_straight[0] keys:', list(ep.keys()))
for k,v in ep.items():
    print(f'  {k}: {str(v)[:200]}')
