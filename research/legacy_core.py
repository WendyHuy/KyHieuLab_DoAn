"""KHL v2: pixel-aspect-aware hand shape and body-relative XY; separate binary masks."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
import numpy as np,cv2
STEPS=32
FEATURES=159
POSE=[0,11,12,13,14,15,16,17,18,19,20,21,22]
PARTS=[(0,65,65),(66,131,131),(132,158,158)]

def extract_video(path,target_fps=12):
 import mediapipe as mp
 cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS)
 if not np.isfinite(fps) or fps<=0:fps=25
 seq=[];times=[];frame_count=0;next_t=0;last_t=-1
 with mp.solutions.holistic.Holistic(static_image_mode=False,model_complexity=0,smooth_landmarks=True,min_detection_confidence=.5,min_tracking_confidence=.5) as model:
  while True:
   ok,frame=cap.read()
   if not ok:break
   t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
   if not np.isfinite(t) or t<=last_t:t=frame_count/fps
   frame_count+=1;last_t=t
   if t+1e-5<next_t:continue
   while next_t<=t+1e-5:next_t+=1/target_fps
   height,width=frame.shape[:2];aspect=width/height
   if max(height,width)>640:frame=cv2.resize(frame,(round(width*640/max(height,width)),round(height*640/max(height,width))))
   r=model.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
   body_ok=False;body=np.zeros((13,2),np.float32);center=np.array([0.,0.]);scale=1.
   if r.pose_landmarks:
    lm=r.pose_landmarks.landmark
    body_ok=lm[11].visibility>.5 and lm[12].visibility>.5
    if body_ok:
     body=np.array([[lm[j].x*aspect,lm[j].y] for j in POSE],np.float32)
     center=(body[1]+body[2])/2;scale=float(np.linalg.norm(body[1]-body[2]));body_ok=scale>.03
   if not body_ok:body=np.zeros((13,2),np.float32);scale=1.
   else:body=(body-center)/scale
   row=[]
   for hand in [r.left_hand_landmarks,r.right_hand_landmarks]:
    values=np.zeros(65,np.float32)
    if hand:
     a=np.array([[v.x*aspect,v.y,v.z*aspect] for v in hand.landmark],np.float32)
     palm=max(float(np.linalg.norm(a[9,:2]-a[0,:2])),.015)
     shape=(a-a[0])/palm
     values[:63]=np.clip(shape.ravel(),-5,5)
     if body_ok:values[63:65]=np.clip((a[0,:2]-center)/scale,-5,5)
    row.extend(values);row.append(float(hand is not None))
   row.extend(np.clip(body.ravel(),-5,5));row.append(float(body_ok))
   seq.append(row);times.append(t)
 cap.release()
 if len(seq)<8:raise ValueError('Video quá ngắn hoặc không đọc được; cần ít nhất 8 khung lấy mẫu.')
 x=np.asarray(seq,np.float32);t=np.asarray(times,np.float64)
 if not np.isfinite(x).all():raise ValueError('Điểm tay không hợp lệ.')
 return x,t,{'decoded_frames':frame_count,'sampled_frames':len(x),'hand_fraction':float(((x[:,65]+x[:,131])>0).mean()),'body_fraction':float(x[:,158].mean()),'duration_observed':float(t[-1]-t[0])}

def resample_masked(seq,times,target_times=None):
 seq=np.asarray(seq,np.float32);times=np.asarray(times,np.float64)
 if len(seq)!=len(times) or len(seq)<2 or np.any(np.diff(times)<=0):raise ValueError('Timestamp phải tăng và khớp số khung.')
 target=np.linspace(times[0],times[-1],STEPS) if target_times is None else np.asarray(target_times)
 hi=np.searchsorted(times,target,side='left').clip(0,len(times)-1);lo=np.maximum(hi-1,0)
 exact=np.abs(times[hi]-target)<1e-6;lo=np.where(exact,hi,lo)
 nearest=np.where(np.abs(times[lo]-target)<=np.abs(times[hi]-target),lo,hi)
 gap=times[hi]-times[lo];ratio=np.divide(target-times[lo],gap,out=np.zeros_like(target),where=gap>0)
 out=np.zeros((len(target),FEATURES),np.float32)
 for begin,end,mask in PARTS:
  both=(seq[lo,mask]>.5)&(seq[hi,mask]>.5)&(gap<=.26)
  out[both,begin:end]=seq[lo[both],begin:end]*(1-ratio[both,None])+seq[hi[both],begin:end]*ratio[both,None]
  out[both,mask]=1
  # Keep a nearby real observation when interpolation crosses a missing interval.
  close=(~both)&(seq[nearest,mask]>.5)&(np.abs(times[nearest]-target)<=.045)
  out[close,begin:end]=seq[nearest[close],begin:end];out[close,mask]=1
 for a,b,mask in [(63,65,65),(129,131,131)]:
  out[:,a:b]*=out[:,158,None]
 return out

def normalize(x,mean,std):
 x=np.asarray(x,np.float32);out=(x-mean)/std
 for begin,end,mask in PARTS:
  out[...,begin:end]*=x[...,mask,None]
  out[...,mask]=x[...,mask]
 for begin,end in [(63,65),(129,131)]:out[...,begin:end]*=x[...,158,None]
 return np.clip(out,-8,8).astype(np.float32)

def temporal_conv(x,w,b,dilation):
 result=np.zeros((len(x),w.shape[-1]),np.float32);size=w.shape[0];pad=dilation*(size-1)//2
 padded=np.pad(x,((pad,pad),(0,0)))
 for k in range(size):result+=padded[k*dilation:k*dilation+len(x)]@w[k]
 return result+b

def infer_logits(sequence,weights,architecture):
 if architecture=='mlp':
  a=sequence.ravel()
  for i in range(3):
   a=a@weights[f'w{i}']+weights[f'b{i}']
   if i<2:a=np.maximum(a,0)
  return a
 if architecture!='tcn':raise ValueError('Kiến trúc không được hỗ trợ.')
 h=np.maximum(sequence@weights['w0']+weights['b0'],0)
 for i,d in [(1,1),(2,2)]:h=np.maximum(temporal_conv(h,weights[f'w{i}'],weights[f'b{i}'],d)+h,0)
 mask=(sequence[:,65]+sequence[:,131])>0
 pooled=np.r_[h[mask].mean(axis=0),h[mask].max(axis=0)] if mask.any() else np.zeros(h.shape[1]*2,np.float32)
 h=np.maximum(pooled@weights['w3']+weights['b3'],0)
 return h@weights['w4']+weights['b4']

def predict(sequence,times,weights,config):
 sampled=resample_masked(sequence,times)
 if ((sampled[:,65]+sampled[:,131])>0).sum()<8: return {'recognized':False,'message':'Chưa thấy đủ bàn tay; chọn lại đoạn ký.','top':[]}
 x=normalize(sampled,np.asarray(config['mean'],np.float32),np.asarray(config['std'],np.float32))
 logits=infer_logits(x,weights,config['architecture']);exp=np.exp(logits-logits.max());p=exp/exp.sum();ids=np.argsort(p)[::-1][:3]
 top=[{'label':config['labels'][int(i)],'score':float(p[i])} for i in ids]
 accepted=bool(p[ids[0]]>=config['threshold'])
 return {'recognized':accepted,'message':top[0]['label'] if accepted else 'Chưa chắc chắn','top':top,'hand_fraction':float(((sampled[:,65]+sampled[:,131])>0).mean()),'warning':'Mô hình thử nghiệm; điểm chưa hiệu chuẩn, chưa kiểm chứng trên người ký mới.'}
