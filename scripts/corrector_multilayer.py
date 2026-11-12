"""只读视觉Hook与独立多层变化缓存接口；不训练、不产生失败标签。"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path


def unwrap_tensor(output):
    value=output[0] if isinstance(output,(tuple,list)) else output
    if getattr(value,'ndim',None)!=3:raise ValueError('视觉层应输出[B,patch,D]，不能静默更换特征定义')
    return value


class ReadOnlyVisionTap:
    """围绕原embed_image调用临时挂Hook。Hook不返回替代值，不重写forward。"""
    def __init__(self,wrapper,layers=(2,5,11),max_view_bytes=32*2**20):
        self.wrapper=wrapper;self.layers=tuple(layers);self.max_view_bytes=max_view_bytes
        if not self.layers or len(set(self.layers))!=len(self.layers) or any(type(i) is not int or i<0 for i in self.layers):
            raise ValueError('视觉层索引须为非负、不重复的整数')
        self.handles=[];self.current=None;self.capture_bytes=0

    def __enter__(self):
        if self.handles:raise RuntimeError('不可重复挂载Hook')
        model=self.wrapper.get_vlm_model();vision=model.vision_model;blocks=vision.encoder.layers
        if self.wrapper.training or any(p.requires_grad for p in self.wrapper.parameters()):raise ValueError('要求独立eval且冻结的视觉实例')
        if max(self.layers)>=len(blocks):raise ValueError('层索引超出真实模型，不自动截断')
        modules={f'block_{i:02d}':blocks[i] for i in self.layers}
        modules.update(vision_postnorm=vision.post_layernorm,connector=model.connector)
        self.expected=set(modules)
        try:
            for name,module in modules.items():self.handles.append(module.register_forward_hook(self._hook(name)))
        except BaseException:
            self.__exit__(None,None,None);raise
        return self

    def _hook(self,name):
        def hook(module,args,output):
            if self.current is None:return None
            if name in self.current:raise RuntimeError('单相机捕获发生重复前向，拒绝混入其他时刻')
            value=unwrap_tensor(output);size=value.numel()*value.element_size()
            if self.capture_bytes+size>self.max_view_bytes:raise RuntimeError('只读Hook内存预算超限')
            self.current[name]=value.detach().clone();self.capture_bytes+=size
            return None  # PyTorch约定：None保留原forward输出。
        return hook

    @contextmanager
    def capture(self,camera):
        if not self.handles or self.current is not None:raise RuntimeError('Hook未挂载或捕获上下文嵌套')
        if not isinstance(camera,str) or not camera:raise ValueError('必须显式指定相机，禁止隐式混合')
        values={};self.current=values;self.capture_bytes=0
        try:
            yield values
            if set(values)!=self.expected:raise RuntimeError('原前向未经过全部指定层/Connector，缓存不完整')
        finally:self.current=None;self.capture_bytes=0

    def __exit__(self,*unused):
        for handle in self.handles:handle.remove()
        self.handles=[];self.current=None


def spatial_mean_pool(features,grid=8):
    """仅为独立新缓存定义的空间均值；不套用末层Connector到中间层。"""
    side=int(features.shape[1]**.5)
    if side*side!=features.shape[1] or grid<1 or side%grid:raise ValueError('patch网格不能按指定网格整除池化')
    b,_,d=features.shape;factor=side//grid
    return features.reshape(b,grid,factor,grid,factor,d).mean(dim=(2,4)).reshape(b,grid*grid,d)


def write_frame(directory,episode,frame_index,timestamp,views,cameras,*,data_fps=10.,grid=8):
    """每帧保存多相机分层特征，训练目标以后按同episode的t+k减t构造。"""
    import numpy as np
    if type(episode) is not int or type(frame_index) is not int or min(episode,frame_index)<0:raise ValueError('episode/frame_index必须有效')
    if abs(timestamp-frame_index/data_fps)>1e-4:raise ValueError('数据帧/时间戳不一致，不以控制Hz替换数据FPS')
    if not cameras or set(views)!=set(cameras) or len(set(cameras))!=len(cameras):raise ValueError('相机集合/顺序不完整')
    names=set(views[cameras[0]])
    if any(set(views[c])!=names for c in cameras):raise ValueError('各相机层集合不一致')
    arrays={};raw_shapes={}
    for name in sorted(names):
        if name=='vision_postnorm':continue  # 仅用于接口核对，不额外重复缓存末层。
        tensors=[spatial_mean_pool(views[c][name],grid) if name.startswith('block_') else views[c][name] for c in cameras]
        raw_shapes[name]=[list(views[c][name].shape) for c in cameras]
        if any(t.shape[0]!=1 for t in tensors):raise ValueError('小规模缓存要求逐帧B=1')
        arrays[name]=np.stack([t.detach().cpu().float().numpy()[0].astype('<f2') for t in tensors])
        if not np.isfinite(arrays[name]).all():raise ValueError('FP16缓存出现非有限值，不缩放修补')
    path=Path(directory)/f'episode_{episode}_frame_{frame_index:06d}.npz'
    if path.exists():raise FileExistsError('不覆盖已有多层帧缓存')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:np.savez_compressed(f,**arrays)
    return {'episode':episode,'frame_index':frame_index,'timestamp':timestamp,'data_fps':data_fps,
        'camera_keys':list(cameras),'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'raw_shapes':raw_shapes,'stored_shapes':{k:list(v.shape) for k,v in arrays.items()},
        'uncompressed_bytes':sum(v.nbytes for v in arrays.values()),'file_bytes':path.stat().st_size,
        'dtype':'float16','pooling':{'type':'nonoverlapping_spatial_mean','grid':grid},
        'feature_sites':'block outputs before vision final LayerNorm; connector after final LayerNorm',
        'target':'same episode: feature[t+k]-feature[t], per layer and per camera; no failure labels'}


def aligned_pairs(records,k_step=10):
    if type(k_step) is not int or k_step<1:raise ValueError('k_step须为正整数')
    index={}
    for i,r in enumerate(records):
        key=(r['episode'],r['frame_index'])
        if key in index:raise ValueError('重复物理帧，不按文件顺序覆盖')
        index[key]=i
    result=[]
    for i,r in enumerate(records):
        j=index.get((r['episode'],r['frame_index']+k_step))
        if j is None:continue
        other=records[j]
        for field in ['data_fps','camera_keys','stored_shapes','dtype','pooling','feature_sites']:
            if r[field]!=other[field]:raise ValueError('配对特征协议不一致')
        if abs(other['timestamp']-r['timestamp']-k_step/r['data_fps'])>1e-4:raise ValueError('视觉变化的实际时间跨度不一致')
        result.append((i,j))
    return result


# Reusable deterministic adapters for the Object10 layer-screening experiment.
# The original read-only tap and per-frame cache functions above are unchanged.
def mean_pool_tokens(x, grid):
    """Pool two camera streams [2, H*W, D] to [2, grid, grid, D]."""
    import numpy as np
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3 or x.shape[0] != 2:
        raise ValueError("expected exactly two separate camera streams")
    side = int(x.shape[1] ** .5)
    if side * side != x.shape[1] or grid < 1 or side % grid:
        raise ValueError("patch token count is not divisible into this grid")
    factor = side // grid
    return x.reshape(2, grid, factor, grid, factor, x.shape[-1]).mean(axis=(2, 4), dtype=np.float32)


def quantize_per_token(x):
    """Reference upstream quantization: per-token int8 and float16 scale."""
    import numpy as np
    x = np.asarray(x, dtype=np.float32)
    scale = np.maximum(np.max(np.abs(x), axis=-1) / 127., 1e-8)
    q = np.clip(np.round(x / scale[..., None]), -127, 127).astype(np.int8)
    return q, scale.astype(np.float16)


def dequantize(q, scale):
    import numpy as np
    return np.asarray(q, dtype=np.float32) * np.asarray(scale, dtype=np.float32)[..., None]
