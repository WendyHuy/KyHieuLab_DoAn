"""Fixed TCN training variant inherited from v3; no test-guided tuning."""
from pathlib import Path
import sys,json
import numpy as np,jax,jax.numpy as jnp
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'data'
import training as v2
from core import resample_masked,normalize,infer_logits,PARTS
LABELS=json.loads((ROOT/'research/protocol.json').read_text(encoding='utf-8'))['labels']
def augment_robust(seq,times,rng):
 increments=rng.uniform(.6,1.4,31);grid=np.r_[0,np.cumsum(increments)];u=grid/grid[-1];u=u**rng.uniform(.8,1.25)
 a,b=rng.uniform(0,.025),rng.uniform(0,.025);target=times[0]+(a+(1-a-b)*u)*(times[-1]-times[0]);x=resample_masked(seq,times,target)
 for begin,end,mask in PARTS:
  seen=x[:,mask]>.5;x[seen,begin:end]+=rng.normal(0,.006,(int(seen.sum()),end-begin))
 for begin in [0,66]:x[:,begin:begin+3]=0
 if rng.random()<.3:
  which=int(rng.integers(0,2));begin,end,mask=PARTS[which];pos=int(rng.integers(0,30));length=int(rng.integers(1,3));x[pos:pos+length,begin:end]=0;x[pos:pos+length,mask]=0
 return x

def fit_candidate(records,seed):
 rng=np.random.default_rng(seed);raw=[];y=[]
 for r in records:
  z=np.load(BASE/'features'/(r['id']+'.npz'));seq=z['features'];times=z['times'];raw.append(resample_masked(seq,times))
  for _ in range(20):raw.append(augment_robust(seq,times,rng))
  y.extend([LABELS.index(r['label'])]*21)
 originals=np.array([raw[i*21] for i in range(len(records))]);mean=originals.mean(axis=(0,1));std=np.maximum(originals.std(axis=(0,1)),.2)
 for _,_,mask in PARTS:mean[mask]=0;std[mask]=1
 x=normalize(np.array(raw),mean,std);y=np.asarray(y,np.int32);by_label={label:np.flatnonzero(y==label) for label in range(len(LABELS))}
 p=v2.initial_params('tcn',len(LABELS),rng);m=jax.tree.map(jnp.zeros_like,p);v=jax.tree.map(jnp.zeros_like,p)
 def loss(params,xx,yy):
  lp=jax.nn.log_softmax(v2.forward(params,xx,'tcn'));ce=-jnp.mean(.98*lp[jnp.arange(len(yy)),yy]+.02*lp.mean(axis=1));return ce+1e-5*sum(jnp.sum(a*a) for k,a in params.items() if k.startswith('w'))
 @jax.jit
 def step(p,m,v,xx,yy,k):
  val,g=jax.value_and_grad(loss)(p,xx,yy);m=jax.tree.map(lambda a,b:.9*a+.1*b,m,g);v=jax.tree.map(lambda a,b:.999*a+.001*b*b,v,g);p=jax.tree.map(lambda a,b,c:a-.001*(b/(1-.9**k))/(jnp.sqrt(c/(1-.999**k))+1e-8),p,m,v);return p,m,v,val
 history=[]
 for k in range(1,501):
  labs=rng.integers(0,len(LABELS),64);ids=np.array([rng.choice(by_label[int(label)]) for label in labs]);p,m,v,val=step(p,m,v,x[ids],y[ids],k)
  if k%50==0:history.append({'step':k,'loss':float(val)})
 p={k:np.asarray(a) for k,a in p.items()};c={'schema':'khl-v2-shape-bodyxy-mask159','architecture':'tcn','steps':32,'features':159,'labels':LABELS,'mean':mean.tolist(),'std':std.tolist(),'threshold':.65,'threshold_calibrated':False,'training_originals':len(records),'seed':seed,'training_steps':500,'parameters':sum(a.size for a in p.values()),'training_variant':'balanced_temporal_jitter_and_missing_hand_augmentation','expert_verified':False,'purpose':'experimental candidate, not independently validated','extractor':'MediaPipe Holistic 0.10.21 model_complexity=0'}
 jp=np.asarray(v2.forward(p,jnp.asarray(x[:2]),'tcn'));npp=np.array([infer_logits(z,p,'tcn') for z in x[:2]]);assert float(np.abs(jp-npp).max())<.001
 return p,c,history
