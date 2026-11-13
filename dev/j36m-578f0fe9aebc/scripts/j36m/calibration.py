"""Fit only independent normal default-camera runs; never held-out/test seeds."""
import numpy as np
from .model import LAYERS


def fit_calibration(trials,checkpoint_sha,manifest_sha,weights=None):
    if len(trials)<2:raise ValueError('at least two independent normal calibration rollouts required')
    ids=set();rows=[]
    for trial in trials:
        key=(trial['condition'],trial['seed'])
        if key in ids:raise ValueError('duplicate calibration case')
        ids.add(key)
        if trial['condition']!='default' or trial['seed'] in (1000,1001,1002) or not trial['normal_success']:
            raise ValueError('calibration must be independent successful normal default rollouts, not test/view5/view10')
        if trial['checkpoint_sha256']!=checkpoint_sha or trial['training_manifest_sha256']!=manifest_sha:
            raise ValueError('calibration provenance mismatch')
        rows.extend(r for r in trial['rows'] if r.get('layer_scores'))
    if len(rows)<30:raise ValueError('insufficient normal score samples')
    if weights is None:
        reliability={}
        for k in LAYERS:
            v=np.asarray([r['layer_scores'][k]['score'] for r in rows if r['layer_scores'][k].get('valid')],dtype=np.float64)
            if len(v)<10:raise ValueError('too few reliable calibration scores: '+k)
            mad=float(np.median(np.abs(v-np.median(v))))
            reliability[k]=(len(v)/len(rows))/(mad+.05)
        total=sum(reliability.values());weights={k:v/total for k,v in reliability.items()}
        weight_fit='normal-only valid coverage / (normal MAD + .05), normalized; heuristic not supervised failure fitting'
    else:weight_fit='explicit prior frozen before test'

    floor={k:max(1e-8,.1*float(np.quantile([min(r['layer_scores'][k]['predicted_rms'],r['layer_scores'][k]['observed_rms'])
                         for r in rows if r['layer_scores'][k].get('valid')],.01))) for k in LAYERS}
    scores=[]
    for r in rows:
        valid=[k for k in LAYERS if r['layer_scores'][k].get('valid') and min(r['layer_scores'][k]['predicted_rms'],r['layer_scores'][k]['observed_rms'])>=floor[k]]
        den=sum(weights[k] for k in valid)
        if den:scores.append(sum(weights[k]*r['layer_scores'][k]['score'] for k in valid)/den)
    if len(scores)<30:raise ValueError('too few valid normal scores after reliability filtering')
    ewma=[]
    for trial in trials:
        previous=None
        for row in trial['rows']:
            if not row.get('layer_scores'):continue
            valid=[k for k in LAYERS if row['layer_scores'][k].get('valid') and min(row['layer_scores'][k]['predicted_rms'],row['layer_scores'][k]['observed_rms'])>=floor[k]]
            den=sum(weights[k] for k in valid)
            if not den:continue
            value=sum(weights[k]*row['layer_scores'][k]['score'] for k in valid)/den
            previous=value if previous is None else .35*value+.65*previous
            ewma.append(previous)
    return {'format':'j36m_calibration_v1','status':'frozen','checkpoint_sha256':checkpoint_sha,
            'training_manifest_sha256':manifest_sha,'weights':weights,'norm_floor':floor,
            'normal_history':ewma,'weight_fit':weight_fit,'cases':[{'condition':c,'seed':s} for c,s in sorted(ids)],
            'state_settings':{'window_size':len(ewma),'base_noise_floor':float(np.quantile(ewma,.95)),
                              'ewma_alpha':.35,'k_on':3.,'k_off':2.,'trigger_margin':.02,
                              'jump_trigger':.06,'trigger_consecutive_steps':5,'reset_consecutive_steps':5,'cooldown_steps':20},
            'threshold_protocol':'fixed normal-history median/MAD; original robust hysteresis/cooldown state, no test refit'}
