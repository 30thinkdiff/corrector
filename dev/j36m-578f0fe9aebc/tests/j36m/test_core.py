import copy,json,sys,tempfile,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from j36m.model import Joint36M,make_model,patchify,unpatchify,reference_pixel_shuffle,LAYERS
from j36m.loss import objective,LossConfig,shared_gradient_audit
from j36m.online import guided_velocity
from j36m.manifest import balanced


def full_inputs():
    return {'block_02':torch.randn(1,2,1024,768),'block_08':torch.randn(1,2,1024,768),
            'connector':torch.randn(1,128,960),'action':torch.randn(1,7)}


class FormalArchitecture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_exact_parameter_budgets(self):
        for key,n in [('J',36793536),('C0',38052800),('S2',37266176),('S8',37266176)]:
            model=make_model(key);self.assertEqual(sum(p.numel() for p in model.parameters()),n);del model
    def test_full_reference_baseline_shapes_and_action_gradients(self):
        for variant,layer in [('C0','connector'),('S2','block_02'),('S8','block_08')]:
            model=make_model(variant).eval().requires_grad_(False);values=full_inputs()
            values['action'].requires_grad_();pred=model(values)['delta_'+layer]
            self.assertEqual(pred.shape,values[layer].shape)
            grad=torch.autograd.grad(pred.square().mean(),values['action'])[0]
            self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.norm()),0)
            del model,pred,grad,values

    def test_patch_roundtrip_official_order_and_camera_separation(self):
        x=torch.arange(2*1024*768).reshape(1,2,1024,768)
        grouped=patchify(x)
        self.assertTrue(torch.equal(x,unpatchify(grouped)))
        for cam in range(2):self.assertTrue(torch.equal(grouped[:,cam*64:(cam+1)*64],reference_pixel_shuffle(x[:,cam])))
        self.assertNotEqual(int(grouped[0,0,0]),int(grouped[0,64,0]))
    def test_real_full_forward_cross_input_and_all_head_gradients(self):
        torch.manual_seed(43);m=Joint36M();x=full_inputs()
        for v in x.values():v.requires_grad_()
        out=m(x)
        for name in LAYERS:self.assertEqual(out['delta_'+name].shape,x[name].shape)
        # A Connector-only objective must backpropagate to BOTH raw streams, not just its own head.
        cross=torch.autograd.grad(out['delta_connector'].square().mean(),[x['block_02'],x['block_08']],retain_graph=True)
        self.assertTrue(all(g.norm()>0 for g in cross))
        reverse=torch.autograd.grad(out['delta_block_02'].square().mean(),x['connector'],retain_graph=True)[0]
        self.assertGreater(float(reverse.norm()),0)
        changed=dict(x);changed['block_02']=x['block_02'].detach().clone();changed['block_02'][0,0,0,0]+=1.
        with torch.no_grad():perturbed=m(changed)['delta_connector']
        self.assertFalse(torch.equal(out['delta_connector'].detach(),perturbed))
        loss,rows=objective(out,{k:torch.randn_like(v) for k,v in out.items()},{k:1. for k in out})
        ga=shared_gradient_audit(rows,m)
        self.assertTrue(all(v>0 for v in ga['norms'].values()))
        loss.backward()
        self.assertGreater(float(x['action'].grad.norm()),0)
        for h in m.heads.values():self.assertTrue(any(p.grad is not None and p.grad.norm()>0 for p in h.parameters()))
        self.assertGreater(float((m.camera[0]-m.camera[1]).norm().detach()),0)
    def test_frozen_real_J_flow_ogg_and_checkpoint_roundtrip(self):
        torch.manual_seed(41);m=Joint36M().eval().requires_grad_(False);x=full_inputs()
        a=x.pop('action');v=torch.randn(1,50,32);xt=torch.randn_like(v)
        targets={k:torch.randn_like(z) for k,z in x.items()}
        cal={'weights':dict(zip(LAYERS,(.25,.25,.5))),'norm_floor':{k:1e-8 for k in LAYERS}}
        before={k:p._version for k,p in m.named_parameters()}
        new,stats=guided_velocity(m,x,xt,v,torch.tensor([.5]),targets,cal)
        self.assertTrue(stats['updated']);self.assertTrue(torch.isfinite(new).all());self.assertGreater(stats['gradient_norm'],0)
        self.assertEqual(before,{k:p._version for k,p in m.named_parameters()})
        self.assertTrue(torch.equal(new[:,1:],v[:,1:]))
        self.assertTrue(torch.equal(new[:,:,7:],v[:,:,7:]))
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            p=Path(d)/'test.pt';torch.save(m.state_dict(),p)
            other=Joint36M().eval();other.load_state_dict(torch.load(p,weights_only=True));
            with torch.no_grad():
                p1=m(dict(x,action=a));p2=other(dict(x,action=a))
            for key in p1:torch.testing.assert_close(p1[key],p2[key],rtol=0,atol=0)
    def test_full_J_single_batch_training_reduces_synthetic_loss(self):
        torch.manual_seed(22);m=Joint36M();x=full_inputs()
        target={'delta_'+k:.1*x[k] for k in LAYERS};scales={k:float(v.square().mean().sqrt()) for k,v in target.items()}
        optimizer=torch.optim.AdamW(m.parameters(),lr=3e-4)
        with torch.no_grad():initial=float(objective(m(x),target,scales)[0])
        for _ in range(6):
            optimizer.zero_grad(set_to_none=True);value=objective(m(x),target,scales)[0]
            value.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        with torch.no_grad():final=float(objective(m(x),target,scales)[0])
        self.assertLess(final,initial*.95)

    def test_train_loss_normalization_and_zero_vector(self):
        p={'delta_'+k:torch.ones(1,2,3)*(i+1) for i,k in enumerate(LAYERS)}
        y={k:torch.zeros_like(v) for k,v in p.items()}
        scales={'delta_'+k:i+1. for i,k in enumerate(LAYERS)}
        loss,rows=objective(p,y,scales)
        self.assertAlmostEqual(float(loss),1.,places=6)
        self.assertTrue(all(float(v['cosine_loss'])==0 for v in rows.values()))
        with self.assertRaises(ValueError):LossConfig(weights=(1,1,1))
    def test_manifest70_is_deterministic_no_metrics_and_exact_budget(self):
        src=json.loads((ROOT/'studies/policy_generalization/corrector/manifests/libero_object10_8695891.json').read_text())
        a,c=balanced(src);src['records'].reverse();b,_=balanced(src)
        self.assertEqual(a['records'],b['records']);self.assertEqual(a['summary']['pairs'],{'train':6791,'validation':2774})
        self.assertEqual(a['summary']['frames'],10265);self.assertEqual(a['summary']['parquet_files'],23)
        for task in range(20,30):
            self.assertEqual(sum(r['task_id']==task and r['split']=='train' for r in a['records']),5)
            self.assertEqual(sum(r['task_id']==task and r['split']=='validation' for r in a['records']),2)
        self.assertEqual(sum('multilayer_export' in r['raw_path'] for r in c['episodes']),5)

if __name__=='__main__':unittest.main()
