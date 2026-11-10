"""Fixed-protocol MLP and temporal convolution experiments. No held-out metrics used for selection."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json,numpy as np,jax,jax.numpy as jnp
from core import resample_masked,normalize,infer_logits,PARTS
ROOT=Path(__file__).resolve().parent/'data'
STEPS=500

def augmentation(seq,times,rng):
 increments=rng.uniform(.85,1.15,31);grid=np.r_[0,np.cumsum(increments)];grid=times[0]+grid/grid[-1]*(times[-1]-times[0])
 x=resample_masked(seq,times,grid)
 for begin,end,mask in PARTS:
  seen=x[:,mask]>.5
  x[seen,begin:end]+=rng.normal(0,.006,(int(seen.sum()),end-begin))
 # Preserve wrist-relative origins and binary validity exactly.
 for begin in [0,66]:x[:,begin:begin+3]=0
 x[:,[63,64,129,130]]*=rng.uniform(.96,1.04)
 return x

def forward(params,x,architecture):
 if architecture=='mlp':
  h=x.reshape((len(x),-1))
  for i in range(3):
   h=h@params[f'w{i}']+params[f'b{i}']
   if i<2:h=jax.nn.relu(h)
  return h
 h=jax.nn.relu(x@params['w0']+params['b0'])
 for i,d in [(1,1),(2,2)]:
  z=jax.lax.conv_general_dilated(h,params[f'w{i}'],(1,),'SAME',rhs_dilation=(d,),dimension_numbers=('NTC','TIO','NTC'))+params[f'b{i}']
  h=jax.nn.relu(z+h)
 mask=((x[:,:,65]+x[:,:,131])>0)[:,:,None]
 avg=jnp.sum(h*mask,axis=1)/jnp.maximum(jnp.sum(mask,axis=1),1)
 maximum=jnp.max(jnp.where(mask,h,-1e8),axis=1)
 maximum=jnp.where(jnp.any(mask,axis=1),maximum,0)
 pooled=jnp.concatenate([avg,maximum],axis=1)
 h=jax.nn.relu(pooled@params['w3']+params['b3'])
 return h@params['w4']+params['b4']

def initial_params(architecture,classes,rng):
 shapes=[(5088,96),(96,48),(48,classes)] if architecture=='mlp' else [(159,48),(3,48,48),(3,48,48),(96,48),(48,classes)]
 p={}
 for i,shape in enumerate(shapes):
  p[f'w{i}']=jnp.asarray(rng.normal(0,np.sqrt(2/np.prod(shape[:-1])),shape),jnp.float32);p[f'b{i}']=jnp.zeros(shape[-1])
 return p

def fit(records,labels,architecture,seed):
 rng=np.random.default_rng(seed);raw=[];y=[]
 for r in records:
  z=np.load(ROOT/'features'/(r['id']+'.npz'));x=z['features'];t=z['times']
  raw.append(resample_masked(x,t))
  for _ in range(20):raw.append(augmentation(x,t,rng))
  y.extend([labels.index(r['label'])]*21)
 originals=np.array([raw[i*21] for i in range(len(records))]);mean=originals.mean(axis=(0,1));std=np.maximum(originals.std(axis=(0,1)),.2)
 for _,_,mask in PARTS:mean[mask]=0;std[mask]=1
 x=normalize(np.array(raw),mean,std);y=np.asarray(y,np.int32)
 p=initial_params(architecture,len(labels),rng);m=jax.tree.map(jnp.zeros_like,p);v=jax.tree.map(jnp.zeros_like,p)
 def loss(params,xx,yy):
  lp=jax.nn.log_softmax(forward(params,xx,architecture))
  # Fixed label smoothing avoids forcing every training clip to a perfect probability.
  ce=-jnp.mean(.98*lp[jnp.arange(len(yy)),yy]+.02*lp.mean(axis=1))
  return ce+1e-5*sum(jnp.sum(a*a) for k,a in params.items() if k.startswith('w'))
 @jax.jit
 def step(p,m,v,xx,yy,k):
  val,g=jax.value_and_grad(loss)(p,xx,yy)
  m=jax.tree.map(lambda a,b:.9*a+.1*b,m,g);v=jax.tree.map(lambda a,b:.999*a+.001*b*b,v,g)
  p=jax.tree.map(lambda a,b,c:a-.001*(b/(1-.9**k))/(jnp.sqrt(c/(1-.999**k))+1e-8),p,m,v)
  return p,m,v,val
 history=[]
 for k in range(1,STEPS+1):
  ids=rng.integers(0,len(x),64);p,m,v,val=step(p,m,v,x[ids],y[ids],k)
  if k%50==0:
   history.append({'step':k,'regularized_training_loss':float(val)});print(architecture,seed,k,float(val),flush=True)
 p={k:np.asarray(a) for k,a in p.items()}
 config={'schema':'khl-v2-shape-bodyxy-mask159','architecture':architecture,'steps':32,'features':159,'labels':labels,'mean':mean.tolist(),'std':std.tolist(),'threshold':.65,'threshold_calibrated':False,'expert_verified':False,'purpose':'user-authorized pilot','training_originals':len(records),'seed':seed,'training_steps':STEPS,'parameters':sum(a.size for a in p.values()),'extractor':'MediaPipe Holistic 0.10.21 model_complexity=0','face_features':False}
 # Compare JAX training path and independent portable NumPy inference.
 test=x[:3];jp=np.asarray(forward(p,jnp.asarray(test),architecture));npp=np.array([infer_logits(z,p,architecture) for z in test]);error=float(np.max(np.abs(jp-npp)))
 assert error<.001,(architecture,error)
 return p,config,history,error

def evaluate(p,c,rows):
 out=[]
 for r in rows:
  z=np.load(ROOT/'features'/(r['id']+'.npz'));a=normalize(resample_masked(z['features'],z['times']),np.array(c['mean'],np.float32),np.array(c['std'],np.float32));logits=infer_logits(a,p,c['architecture']);v=np.exp(logits-logits.max());v=v/v.sum();i=int(v.argmax())
  out.append({'id':r['id'],'provided':r['label'],'predicted':c['labels'][i],'score':float(v[i]),'correct':c['labels'][i]==r['label']})
 return out
