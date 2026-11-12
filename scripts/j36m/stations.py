"""Additional read-only hooks for real-frame connector scale attribution.

No forward replacement. Comparisons are within the same named site across time,
never elementwise subtraction between different feature spaces.
"""
import hashlib,inspect
from contextlib import contextmanager
import numpy as np
import torch
from corrector_multilayer import ReadOnlyVisionTap,quantize_per_token
from .model import patchify,reference_pixel_shuffle


def compare(a,b):
    a=np.asarray(a,dtype=np.float64).reshape(-1);b=np.asarray(b,dtype=np.float64).reshape(-1)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('comparison shape/numerics')
    na,nb=np.linalg.norm(a),np.linalg.norm(b)
    return {'mse':float(np.square(a-b).mean()),'relative_l2':float(np.linalg.norm(a-b)/max(nb,1e-12)),
            'cosine':float(a@b/(na*nb)) if na>1e-12 and nb>1e-12 else None}


@torch.no_grad()
def capture_sites(wrapper,images):
    if len(images)!=2:raise ValueError('two cameras')
    connector=wrapper.get_vlm_model().connector;post_shuffle=[];views=[]
    def pre(module,args):post_shuffle.append(args[0].detach().clone())
    handle=connector.modality_projection.register_forward_pre_hook(pre)
    try:
        with ReadOnlyVisionTap(wrapper,(2,8,9,10,11),max_view_bytes=48*2**20) as tap:
            for i,image in enumerate(images):
                with tap.capture(f'camera{i+1}') as out:wrapper.embed_image(image)
                if len(post_shuffle)!=i+1:raise RuntimeError('ambiguous connector forward')
                expected=reference_pixel_shuffle(out['vision_postnorm'],4)
                actual=connector.pixel_shuffle(out['vision_postnorm'],connector.scale_factor)
                if connector.scale_factor!=4 or not torch.equal(expected,actual) or not torch.equal(actual,post_shuffle[i]):
                    raise ValueError('installed pixel-shuffle ordering does not match formal grouping')
                views.append(dict(out,pixel_shuffle=post_shuffle[i]))
    finally:handle.remove()
    return {name:torch.stack([v[name].float() for v in views],1).cpu().numpy() for name in views[0]}


def audit_pair(wrapper,images0,images10,old0,old10,*,episode,t,policy_revision,weight_sha,raw_cache=None):
    a,b=capture_sites(wrapper,images0),capture_sites(wrapper,images10)
    sites={}
    for name,x in a.items():
        y=b[name];delta=y-x
        sites[name]={'shape':list(x.shape),'rms':float(np.square(x,dtype=np.float64).mean()**.5),
                     'delta_rms':float(np.square(delta,dtype=np.float64).mean()**.5)}
    # A pure shuffle preserves every value, norm and every same-site temporal delta.
    if not np.isclose(sites['vision_postnorm']['rms'],sites['pixel_shuffle']['rms'],rtol=1e-7):raise RuntimeError('shuffle changed values')
    chain=('block_08','block_09','block_10','block_11','vision_postnorm','pixel_shuffle','connector')
    gains={left+'->'+right:{kind:sites[right][kind]/sites[left][kind] if sites[left][kind]>1e-12 else None for kind in ('rms','delta_rms')} for left,right in zip(chain,chain[1:])}
    deep0=a['connector'].reshape(128,960);deep10=b['connector'].reshape(128,960)
    requantized=[]
    for value,stored in [(deep0,old0),(deep10,old10)]:
        q,scale=quantize_per_token(value)
        reconstructed=q.astype('float32')*scale.astype('float32')[:,None]
        requantized.append(bool(np.array_equal(reconstructed,stored)))
    if not all(requantized):raise ValueError('recomputed Connector quantization differs from frozen cache')
    if raw_cache:
        for values,cached in zip((a,b),raw_cache,strict=True):
            for k in ('block_02','block_08'):
                if not np.array_equal(values[k][0].astype('float16'),cached[k]):raise ValueError('raw feature hook/camera/frame mismatch')
    return {'episode':episode,'t':t,'future_t':t+10,'policy_revision':policy_revision,'weight_sha256':weight_sha,
            'installed_shuffle_source_sha256':hashlib.sha256(inspect.getsource(type(wrapper.get_vlm_model().connector).pixel_shuffle).encode()).hexdigest(),
            'sites':sites,'amplitude_ratios_only_no_cross_site_element_subtraction':gains,'quantization_level':compare(deep0,old0),
            'quantization_delta':compare(deep10-deep0,old10-old0),
            'requantized_connector_matches':requantized,'same_frame_same_camera_order':True,'pixel_shuffle_layout_verified':True}


def run_station_audit(args,cfg,records,guard):
    """Explicit future GPU command, never invoked by CPU scale audit."""
    import os,sys,json
    from pathlib import Path
    from corrector_data_workbench import cached_policy_snapshot,cached_visual_extractor,decode_rgb,digest,write_json
    from multilayer_mvp.data import POLICY,WEIGHT,IMAGES
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    guard();os.environ.update(HF_HOME=str(args.runtime/'hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    if not (args.source_root/'src/siglip_dynamics/extract.py').is_file():raise FileNotFoundError('frozen extractor absent')
    sys.path.insert(0,str(args.source_root/'src'));sys.path.insert(0,str(args.runtime/'lerobot/src'))
    from siglip_dynamics.extract import SmolVLAExtractor
    import pyarrow.parquet as pq
    snapshot=Path(cached_policy_snapshot(False))
    if digest(snapshot/'model.safetensors')!=WEIGHT:raise ValueError('policy weights changed')
    encoder=cached_visual_extractor(SmolVLAExtractor,snapshot,'cuda',allow_network=False)
    encoder.policy.eval().requires_grad_(False);wrapper=encoder.model.vlm_with_expert
    output.mkdir(parents=True);rows=[]
    for record in records:
        ep=record['spec']['episode'];old=Path(cfg['features_root'])/f'episode_{ep}'
        metadata=json.loads((old/'metadata.json').read_text());sources=metadata['raw_provenance']['sources']
        if len(sources)!=1:raise ValueError('station audit requires single explicit source shard')
        path=args.data_root/sources[0]['file']
        if not path.is_file() or digest(path)!=sources[0]['sha256']:raise ValueError('source shard SHA/availability')
        table=pq.read_table(path,columns=['episode_index','frame_index',*IMAGES]).to_pylist()
        selected={r['frame_index']:r for r in table if r['episode_index']==ep and r['frame_index'] in (0,10)}
        if set(selected)!={0,10}:raise ValueError('missing exact aligned source frames')
        images=[[encoder._prepare_image(decode_rgb(selected[t][k]['bytes'])) for k in IMAGES] for t in (0,10)]
        a=record['arrays'];old_values=[a['z_q'][t].astype('float32')*a['z_scale'][t].astype('float32')[:,None] for t in (0,10)]
        rows.append(audit_pair(wrapper,images[0],images[1],old_values[0],old_values[1],episode=ep,t=0,policy_revision=POLICY,weight_sha=WEIGHT,
                    raw_cache=[{k:a[k][t] for k in ('block_02','block_08')} for t in (0,10)]))
    report={'status':'passed','source':[r['provenance'] for r in records],'pairs':rows,
            'note':'same frames, same cameras/weights; shuffle verified numerically, not assumed'}
    write_json(output/'station_audit.json',report);return report
