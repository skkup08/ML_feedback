

"""Fitness research baseline. Output path is supplied by the Colab wrapper.
Generated 2026-10-24. Dataset approval and unknown labels are never bypassed.
Run by the accompanying single-cell Colab notebook. Not a clinical safety system.
"""
from __future__ import annotations
import os
# Must precede Matplotlib/MediaPipe imports; Colab's inline backend is not in this venv.
os.environ["MPLBACKEND"] = "Agg"
import argparse, hashlib, io, json, re, shutil, sys, tempfile, time, traceback, zipfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse
import cv2
import joblib
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

SEED = 42
BODY = np.array([11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32])
UPPER = np.array([11,12,13,14,15,16,23,24])
NAMES = {11:'left_shoulder',12:'right_shoulder',13:'left_elbow',14:'right_elbow',
 15:'left_wrist',16:'right_wrist',23:'left_hip',24:'right_hip',25:'left_knee',
 26:'right_knee',27:'left_ankle',28:'right_ankle',29:'left_heel',30:'right_heel',
 31:'left_foot_index',32:'right_foot_index'}
ANGLES = [(11,13,15),(12,14,16),(13,11,23),(14,12,24),
          (11,23,25),(12,24,26),(23,25,27),(24,26,28)]
PATTERNS = {
 'side_bend':r'\b(abs|abdominals?|side\s*bends?)\b',
 'bent_over_row':r'\b(back|rows?|rowing)\b',
 'bicep_curl':r'\b(biceps?|bicep\s*curls?|curls?)\b',
 'push_up':r'\b(chest|push\s*ups?)\b',
 'shoulder_press':r'\b(shoulders?|shoulder\s*press)\b',
 'squat':r'\b(legs?|squats?)\b',
 'triceps_extension':r'\b(triceps?|tricep\s*extensions?)\b'}
EX_KO = {'side_bend':'덤벨 사이드 벤드','bent_over_row':'벤트오버 로우',
 'bicep_curl':'바이셉 컬','push_up':'푸시업','shoulder_press':'숄더 프레스',
 'squat':'스쿼트','triceps_extension':'트라이셉스 익스텐션','plank':'플랭크',
 'bench_press':'벤치프레스','lunge':'런지','pull_up':'풀업',
 'handstand_push_up':'물구나무 푸시업','wall_push_up':'벽 푸시업',
 'jumping_jack':'점핑 잭'}
UCF_MAP = {'BenchPress':'bench_press','BodyWeightSquats':'squat','Lunges':'lunge',
 'PullUps':'pull_up','PushUps':'push_up','HandstandPushups':'handstand_push_up',
 'WallPushups':'wall_push_up','JumpingJack':'jumping_jack'}
MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
             'pose_landmarker_full/float16/1/pose_landmarker_full.task')
MD_PAGE = 'https://data.mendeley.com/datasets/kgbb3yn47p/3'
GH_RAW = 'https://raw.githubusercontent.com/NgoQuocBao1010/Exercise-Correction/'
GH_REPO = 'https://github.com/NgoQuocBao1010/Exercise-Correction'
FEATURE_VERSION = 'fitness_consolidated_v4_world1232_image64_summary256'
# These are fixed, non-diagnostic coaching prompts, NOT an independently trained LLM.
CORRECTIONS = {
 'plank_hips_high': {'exercise':'plank','name':'플랭크에서 골반이 높게 들린 패턴',
  'tip':'옆모습을 확인하며 골반을 조금 낮춰 어깨–골반–발목이 대체로 이어지게 해보세요. 유지가 어려우면 버티는 시간을 줄이세요.'},
 'plank_hips_low': {'exercise':'plank','name':'플랭크에서 골반이 처진 패턴',
  'tip':'골반을 조금 들어 몸통의 처짐을 줄이고 복부에 가볍게 힘을 주세요. 자세를 유지하기 어렵다면 무릎을 지지하는 쉬운 변형으로 바꿔보세요.'},
 'curl_torso_lean_back': {'exercise':'bicep_curl','name':'컬에서 상체를 뒤로 젖히는 패턴',
  'tip':'무게나 반복 수를 줄이고 상체의 반동 없이 천천히 들어보세요. 몸통을 편안하게 세운 상태에서 팔꿈치 위치가 크게 움직이지 않는지 확인하세요.'},
 # The following are enabled ONLY when an explicit reviewed manifest provides labels.
 'squat_knee_collapse': {'exercise':'squat','name':'스쿼트에서 무릎이 안쪽으로 무너지는 패턴',
  'tip':'부하와 깊이를 줄여 발과 무릎의 진행 방향을 함께 확인하세요. 억지로 무릎을 벌리지 말고 편안한 범위에서 다시 촬영하세요.'},
 'squat_heel_lift': {'exercise':'squat','name':'스쿼트에서 뒤꿈치가 들리는 패턴',
  'tip':'부하나 깊이를 줄여 발의 지지가 유지되는 범위를 확인하세요. 발 간격과 신발 조건도 함께 점검하세요.'},
 'row_back_rounding': {'exercise':'bent_over_row','name':'로우에서 등 모양이 크게 변하는 패턴',
  'tip':'무게를 줄이고 엉덩이를 접어 몸통을 지지한 상태에서 반복해보세요. 촬영 각도를 바꿔 실제 자세인지 먼저 확인하세요.'},
 'press_back_arch': {'exercise':'shoulder_press','name':'프레스에서 몸통을 과하게 젖히는 패턴',
  'tip':'부하와 가동 범위를 줄이고 몸통을 편안하게 지지한 상태로 다시 시도하세요. 통증 없는 범위를 우선하세요.'}}
CHECKLIST = {
 'squat':'발의 지지가 유지되는지, 무릎의 진행 방향과 몸통의 흔들림을 정면·측면에서 확인하세요.',
 'push_up':'어깨–골반–발목의 연결과 좌우 팔의 움직임을 확인하세요. 어렵다면 쉬운 변형으로 바꾸세요.',
 'bicep_curl':'몸통 반동과 팔꿈치 위치 변화를 확인하고, 무거우면 부하를 줄이세요.',
 'shoulder_press':'몸통을 과하게 젖히는지와 좌우 팔의 움직임을 확인하세요.',
 'bent_over_row':'몸통 지지, 반동, 좌우 팔 움직임을 확인하고 무거우면 부하를 줄이세요.',
 'side_bend':'골반과 몸통의 불필요한 회전, 빠른 반동이 있는지 확인하세요.',
 'triceps_extension':'몸통 반동과 팔꿈치 위치 변화를 확인하고 편안한 가동 범위에서 움직이세요.',
 'plank':'골반이 높게 들리거나 처지지 않는지 옆모습으로 확인하세요.'}


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str),encoding='utf-8')
    temp.replace(path)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()


class Downloader:
    def __init__(self, root, limit_gb):
        self.root=Path(root); self.limit=int(limit_gb*1024**3); self.used=0
        self.s=requests.Session()
        self.s.headers.update({'User-Agent':'Mozilla/5.0 FitnessResearch/1.0',
                               'Accept-Encoding':'identity'})
        self.s.mount('https://', HTTPAdapter(max_retries=Retry(
            total=2,backoff_factor=1,status_forcelist=[429,500,502,503,504])))
    def json(self,url):
        r=self.s.get(url, timeout=(20,60)); r.raise_for_status(); return r.json()
    def get(self,url,path,size=0,sha=None):
        if urlparse(url).scheme not in {'http','https'}:
            raise ValueError('HTTP(S) 다운로드 URL만 허용합니다.')
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        part=path.with_name(path.name+'.part'); error=None
        for attempt in range(3):
            try:
                start=part.stat().st_size if part.exists() else 0
                headers={'Range':f'bytes={start}-'} if start else {}
                with self.s.get(url,headers=headers,stream=True,timeout=(25,120)) as r:
                    if r.status_code==416:
                        part.unlink(missing_ok=True); raise ValueError('이어받기 범위를 초기화했습니다.')
                    r.raise_for_status()
                    if 'text/html' in r.headers.get('Content-Type','').lower():
                        raise ValueError('실제 파일 대신 로그인/오류 HTML을 받았습니다.')
                    if r.status_code==206:
                        if not r.headers.get('Content-Range','').startswith(f'bytes {start}-'):
                            raise ValueError('잘못된 이어받기 응답')
                    else: start=0
                    length=int(r.headers.get('Content-Length',0) or 0)
                    total=int(size or (start+length))
                    remain=max(total-start,0)
                    if remain+self.used>self.limit: raise RuntimeError('전체 다운로드 상한 초과: MAX_DOWNLOAD_GB를 확인하세요.')
                    if remain+1024**3>shutil.disk_usage(self.root).free:
                        raise RuntimeError('임시 디스크 공간 부족')
                    with part.open('ab' if start else 'wb') as f, tqdm(
                        total=total or None,initial=start,unit='B',unit_scale=True,
                        desc=path.name,mininterval=2,file=sys.stdout) as bar:
                        for block in r.iter_content(2*1024*1024):
                            if not block: continue
                            if self.used+len(block)>self.limit: raise RuntimeError('전체 다운로드 상한 초과')
                            f.write(block); self.used+=len(block); bar.update(len(block))
                    if total and part.stat().st_size!=total: raise IOError('파일 크기 불일치')
                if sha and sha256(part).lower()!=sha.lower():
                    part.unlink(missing_ok=True); raise ValueError('SHA256 불일치')
                part.replace(path); return path
            except requests.exceptions.SSLError as e:
                raise RuntimeError('HTTPS 인증서 확인 실패. 검증을 끄지 않고 이 소스를 중단합니다: '+str(e)) from e
            except RuntimeError: raise
            except Exception as e:
                error=e
                if attempt<2: print('다운로드 재시도:',type(e).__name__,flush=True);time.sleep(2)
        raise RuntimeError(f'다운로드 실패: {url}\n{error}')


def file_records(obj):
    out=[]
    if isinstance(obj,dict):
        c=obj.get('content_details') or {}
        n=obj.get('filename') or obj.get('name'); u=c.get('download_url') or obj.get('download_url')
        if n and u:
            out.append({'name':n,'url':u,'size':int(obj.get('size') or c.get('size') or 0),
                        'sha':c.get('sha256_hash')})
        for v in obj.values():out.extend(file_records(v))
    elif isinstance(obj,list):
        for v in obj:out.extend(file_records(v))
    return out


def parse_mendeley(name):
    p=Path(name); ex=q=subject=None
    for piece in [p.stem]+list(reversed(p.parts[:-1])):
        text=re.sub(r'[^a-z0-9]+',' ',piece.lower())
        if q is None:
            hits=set(re.findall(r'\b(good|bad)\b',text))
            if len(hits)==1:q=hits.pop()
        if ex is None:
            hits=[k for k,v in PATTERNS.items() if re.search(v,text)]
            if len(hits)==1:ex=hits[0]
        if subject is None:
            m=re.search(r'\b(?:subject|participant|person|sub|s|p)\s*(?:id\s*)?0*(\d+)\b',text)
            if m:subject=f'S{int(m.group(1)):03d}'
    if subject is None:
        m=re.match(r'^0*(\d+)[_\s-]',p.stem)
        if m:subject=f'S{int(m.group(1)):03d}'
    if subject is None:
        for i,piece in enumerate(p.parts[:-2]):
            if re.sub(r'[^a-z]','',piece.lower()) in {'datasetsubjectwise','subjectwise','datasetsubjects','subjects'}:
                v=p.parts[i+1]
                subject=f'S{int(v):03d}' if v.isdigit() else 'person:'+v.strip().lower()
                break
    if not all((ex,q,subject)):
        raise ValueError(f'운동/Good-Bad/사람 ID 해석 불가: {name}')
    return ex,q,'mendeley:'+subject


def frame_features(points, selected=BODY):
    """MediaPipe IMAGE xyz, not world xyz. Facial landmarks are excluded.
    Same transformation is used for repository CSVs and newly processed video.
    Image y/z conventions are retained; these are not physical joint angles.
    Returns one feature row and validity flag per input frame.
    """
    a=np.asarray(points,dtype=np.float32)
    if a.ndim==2:a=a[None]
    xyz=a[:,:,:3].copy(); vis=a[:,:,3]
    hip=(xyz[:,23]+xyz[:,24])/2;shoulder=(xyz[:,11]+xyz[:,12])/2
    scale=np.linalg.norm((shoulder-hip)[:,:2],axis=1)
    core=np.isfinite(xyz[:,[11,12,23,24]]).all(axis=(1,2))
    core &= (np.maximum(vis[:,11],vis[:,12])>=.35)&(np.maximum(vis[:,23],vis[:,24])>=.35)
    valid=core&(scale>.015)&np.isfinite(scale)
    norm=(xyz[:,selected]-hip[:,None,:])/np.maximum(scale[:,None,None],.015)
    visible=(vis[:,selected]>=.30)&np.isfinite(norm).all(axis=2)
    valid &= (visible.mean(axis=1)>=.4)
    norm=np.where(visible[:,:,None],np.clip(norm,-8,8),0)
    feats=np.concatenate([norm.reshape(len(a),-1),visible.astype(np.float32)],axis=1)
    return np.nan_to_num(feats).astype(np.float32),valid


def temporal_features(world,visibility,times):
    x=np.asarray(world,dtype=np.float32)[:,BODY,:].copy(); t=np.asarray(times)
    v=np.asarray(visibility)[:,BODY];x[v<.30]=np.nan
    core=np.isfinite(x[:,[0,1,6,7]]).all(axis=(1,2))
    if core.sum()<12 or core.mean()<.55:raise ValueError('시간 특징용 몸통 검출 부족')
    flat=x.reshape(len(x),-1)
    for j in range(flat.shape[1]):
        ok=np.isfinite(flat[:,j])
        if ok.sum()<max(3,.20*len(x)):raise ValueError('가려진 관절이 너무 많음')
        flat[:,j]=np.interp(t,t[ok],flat[ok,j])
    x=flat.reshape(-1,16,3)
    hip=(x[:,6]+x[:,7])/2;shoulder=(x[:,0]+x[:,1])/2
    scale=np.median(np.linalg.norm(shoulder-hip,axis=1))
    if scale<.03:raise ValueError('몸통 크기 추정 실패')
    x=np.clip((x-hip[:,None])/scale,-8,8)
    idx={int(j):i for i,j in enumerate(BODY)}; angles=[]
    for a,b,c in ANGLES:
        u=x[:,idx[a]]-x[:,idx[b]];v=x[:,idx[c]]-x[:,idx[b]]
        cos=(u*v).sum(1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)+1e-6)
        angles.append(np.arccos(np.clip(cos,-1,1))/np.pi)
    f=np.concatenate([x.reshape(len(x),-1),np.array(angles).T],axis=1)
    vel=np.clip(np.diff(f,axis=0)/np.maximum(np.diff(t)[:,None],1e-3),-20,20)
    grid=np.linspace(t[0],t[-1],16)
    order=np.stack([np.interp(grid,t,f[:,j]) for j in range(f.shape[1])],axis=1)
    result=np.concatenate([f.mean(0),f.std(0),np.quantile(f,.1,axis=0),
      np.quantile(f,.9,axis=0),np.abs(vel).mean(0),vel.std(0),order.ravel()]).astype(np.float32)
    if not np.isfinite(result).all():raise ValueError('시간 특징에 NaN/Inf')
    return result


def image_summary(frame_rows):
    """Video-level image-pose aggregate for clips whose world joints are occluded."""
    rows=np.asarray(frame_rows,dtype=np.float32)
    if rows.ndim!=2 or len(rows)<1 or not np.isfinite(rows).all():
        raise ValueError('유효한 영상 요약 특징이 없습니다.')
    return np.concatenate([rows.mean(0),rows.std(0),
        np.quantile(rows,.1,axis=0),np.quantile(rows,.9,axis=0)]).astype(np.float32)


def atomic_model_save(value,path):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    joblib.dump(value,tmp,compress=3)
    tmp.replace(path)


class PoseEnvironmentError(RuntimeError):
    """A shared runtime/model initialization error, not a bad input video."""


def extract_pose(path,asset,cfg):
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except Exception as e:
        raise PoseEnvironmentError("MediaPipe import 실패: " + repr(e)) from e
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise ValueError('영상 열기 실패')
    fps,count=cap.get(cv2.CAP_PROP_FPS),cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not np.isfinite([fps,count]).all() or fps<=0 or count<12:
        cap.release();raise ValueError('유효한 FPS/프레임 수가 없음')
    count=int(count); n=min(count,int(cfg['MAX_FRAMES']),max(12,int(count/fps*cfg['SAMPLE_FPS'])))
    wanted=set(np.linspace(0,count-1,n).astype(int).tolist())
    opts=vision.PoseLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=str(asset),delegate=mp_python.BaseOptions.Delegate.CPU),
      running_mode=vision.RunningMode.VIDEO,num_poses=2,min_pose_detection_confidence=.5,
      min_pose_presence_confidence=.5,min_tracking_confidence=.5,output_segmentation_masks=False)
    image_points=[];world=[];times=[];many=0;last_ts=-1
    try:
        try:
            detector=vision.PoseLandmarker.create_from_options(opts)
        except Exception as e:
            raise PoseEnvironmentError("관절 모델 초기화 실패: " + repr(e)) from e
        with detector as pose:
            for i in range(count):
                if not cap.grab():break
                if i not in wanted:continue
                ok,img=cap.retrieve()
                if not ok:continue
                h,w=img.shape[:2]
                if max(h,w)>640:img=cv2.resize(img,(max(1,round(w*640/max(h,w))),max(1,round(h*640/max(h,w)))))
                rgb=np.ascontiguousarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
                ts=max(last_ts+1,round(i/fps*1000));last_ts=ts
                result=pose.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),ts)
                im=np.full((33,4),np.nan,dtype=np.float32);im[:,3]=0
                wo=np.full((33,3),np.nan,dtype=np.float32)
                # Never silently track a different person in a multi-person scene.
                if len(result.pose_landmarks)>1:many+=1
                elif len(result.pose_landmarks)==1 and result.pose_world_landmarks:
                    im=np.array([[p.x,p.y,p.z,p.visibility] for p in result.pose_landmarks[0]],np.float32)
                    wo=np.array([[p.x,p.y,p.z] for p in result.pose_world_landmarks[0]],np.float32)
                image_points.append(im);world.append(wo);times.append(i/fps)
    finally:cap.release()
    if len(times)<12 or len(times)<.8*n:raise ValueError('디코딩된 프레임 부족')
    if many/len(times)>.15:raise ValueError('여러 사람이 보여 운동 대상자를 확정할 수 없음')
    return np.array(image_points),np.array(world),np.array(times,np.float32)


def pose_preflight(asset,work,cfg):
    """Test import, backend, video decoding and actual CPU task execution before dataset downloads.
    The generated blank clip is a technical smoke test, not an accuracy benchmark.
    """
    probe=Path(work)/'_runtime_smoke_test.avi'
    writer=None
    try:
        import matplotlib
        import mediapipe as mp
        matplotlib.use('Agg',force=True)
        writer=cv2.VideoWriter(str(probe),cv2.VideoWriter_fourcc(*'MJPG'),12.0,(256,256))
        if not writer.isOpened():
            raise RuntimeError('사전 검사 영상 생성 실패: OpenCV MJPG codec')
        for _ in range(16):writer.write(np.zeros((256,256,3),np.uint8))
        writer.release();writer=None
        points,world,t=extract_pose(probe,asset,cfg)
        if len(t)<12 or points.shape[1:]!=(33,4):
            raise RuntimeError('사전 검사 영상의 프레임 처리 결과가 올바르지 않습니다.')
        return {'status':'passed','matplotlib_backend':str(matplotlib.get_backend()),
                'mediapipe_version':mp.__version__,'opencv_version':cv2.__version__,
                'decoded_and_inferred_frames':len(t),
                'note':'합성 빈 영상으로 초기화/디코딩/실제 추론만 확인. 사람 검출 성능 검증 아님.'}
    except Exception as e:
        raise PoseEnvironmentError('대용량 데이터 다운로드 전 환경 검사 실패: '+repr(e)) from e
    finally:
        if writer is not None:writer.release()
        probe.unlink(missing_ok=True)


def group_split(records):
    """Per-source subject/origin groups; never random-split frames."""
    result={}; by_source=defaultdict(list)
    for i,r in enumerate(records):by_source[r['source']].append(i)
    for source,ids in by_source.items():
        fixed=[records[i].get('split','') for i in ids]
        if all(x in {'train','validation','test'} for x in fixed):
            for i,x in zip(ids,fixed):result[i]=x
            continue
        # Mixed fixed/non-fixed splits are ambiguous and intentionally rejected.
        if any(fixed):raise ValueError(f'{source}: 고정 split은 모든 행에 지정하거나 모두 비워야 합니다.')
        g=np.array([records[i]['group'] for i in ids]);y=np.array([records[i]['exercise']+'|'+records[i].get('quality','') for i in ids])
        if len(set(g))<5:
            for i in ids:result[i]='train'
            print(f'{source}: 그룹이 5개 미만, 독립 평가 없음.',flush=True);continue
        chosen=None
        for seed in range(SEED,SEED+100):
            tv,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(y,y,g))
            a,b=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed+500).split(y[tv],y[tv],g[tv]))
            tr,va=tv[a],tv[b]
            if set(y[tr])==set(y):chosen=(tr,va,te);break
        if chosen is None:raise ValueError(f'{source}: 그룹 단위 학습 분할에 모든 운동을 넣을 수 없습니다.')
        for part,ix in zip(('train','validation','test'),chosen):
            for j in ix:result[ids[int(j)]]=part
    groups=defaultdict(set)
    for i,r in enumerate(records):groups[(r['source'],r['group'])].add(result[i])
    if any(len(v)>1 for v in groups.values()):raise ValueError('동일 사람/원본 그룹이 split 간 중복됨')
    return [result[i] for i in range(len(records))]


def make_model(trees=250):
    return ExtraTreesClassifier(n_estimators=trees,min_samples_leaf=2,max_features='sqrt',
      class_weight='balanced',n_jobs=-1,random_state=SEED)


def metrics(y,p):
    if not len(y):return {'n':0,'note':'평가 자료 없음'}
    return {'n':len(y),'accuracy':float(accuracy_score(y,p)),
      'macro_f1':float(f1_score(y,p,average='macro',zero_division=0)),
      'classes':classification_report(y,p,output_dict=True,zero_division=0)}


class Pipeline:
    def __init__(self,cfg):
        self.cfg=cfg;self.out=Path(cfg['OUT']);self.work=Path(cfg['WORK'])
        self.out.mkdir(parents=True,exist_ok=True);self.work.mkdir(parents=True,exist_ok=True)
        self.dl=Downloader(self.work,cfg['MAX_DOWNLOAD_GB'])
        self.records=[];self.features=[];self.frames=[];self.summaries=[];self.seen={};self.failed=[];self.status=[]
        self.asset=self.out/'pose_landmarker_full.task';self.git_revision='main'
        self.errors={};self.csv_data={};self.eval={}
        cv2.setNumThreads(1)
    def log(self,source,status,detail,**extra):
        row={'source':source,'status':status,'detail':str(detail),**extra}
        self.status.append(row);dump(self.out/'source_status.json',self.status)
        print(f'[{source}] {status}: {detail}',flush=True)
    def add_video(self,path,rec):
        try:
            sha=sha256(path)
            if sha in self.seen:
                old=self.seen[sha]
                if old['exercise']!=rec['exercise'] or (old.get('quality') and rec.get('quality') and old['quality']!=rec['quality']):
                    raise ValueError('동일 영상의 라벨 충돌')
                self.failed.append({**rec,'reason':'동일 SHA256 영상 중복 제외'});return
            points,world,t=extract_pose(path,self.asset,self.cfg)
            ff,valid=frame_features(points)
            if valid.sum()<12 or valid.mean()<.55:raise ValueError('사용 가능한 관절 프레임 부족')
            ix=np.where(valid)[0]
            ix=ix[np.linspace(0,len(ix)-1,min(16,len(ix))).astype(int)]
            try:vec=temporal_features(world,points[:,:,3],t);treason=''
            except ValueError as e:vec=None;treason=str(e)
            self.seen[sha]=rec
            self.records.append({**rec,'sha256':sha,'sampled_frames':len(t),
              'usable_frame_ratio':float(valid.mean()),'temporal_available':vec is not None,
              'temporal_exclusion':treason})
            self.features.append(vec);self.frames.append(ff[ix]);self.summaries.append(image_summary(ff[valid]))
        except PoseEnvironmentError as e:
            self.failed.append({**rec,'reason':str(e)})
            self.save_failures()
            self.log(rec['source'],'fatal_environment',e)
            raise
        except Exception as e:
            self.failed.append({**rec,'reason':str(e)})
            # Show concrete causes immediately, rather than hiding 1,000 failures.
            if len(self.failed)<=3:
                print('영상 제외:',rec.get('name',''),'=>',str(e),flush=True)
                self.save_failures()
        else:
            if len(self.records)==1:
                print('첫 영상 관절 추출 성공:',rec.get('name',''),
                      '| 프레임:',len(t),'| 시간 특징:',vec is not None,flush=True)
    def progress(self,stage):
        dump(self.out/'progress.json',{'stage':stage,'usable_videos':len(self.records),
            'excluded_videos':len(self.failed),'downloaded_GiB':self.dl.used/1024**3,
            'by_source':dict(Counter(r['source'] for r in self.records)),
            'note':'진행 기록이며 중단 지점 재개용 특징 캐시가 아닙니다.'})
    def checkpoint(self,bundle,stage):
        value=dict(bundle)
        value.update({'feature_version':FEATURE_VERSION,
            'config':{k:v for k,v in self.cfg.items() if k not in {'WORK','EXTRA_MANIFEST'}},
            'corrections':CORRECTIONS,'checkpoint_stage':stage,
            'score_note':'분류 점수는 보정된 정확도/안전 확률이 아닙니다.'})
        atomic_model_save(value,self.out/'fitness_models.joblib')
        dump(self.out/'metrics.json',self.eval)
        self.progress(stage)
    def save_failures(self):
        columns=['source','name','exercise','quality','group','split','errors','reason']
        pd.DataFrame(self.failed,columns=columns).to_csv(
            self.out/'failed_or_skipped_videos.csv',index=False,encoding='utf-8-sig')
    def mendeley(self):
        direct=self.cfg.get('MENDELEY_ZIP_URL','').strip();record=None
        if direct:record={'url':direct,'name':'mendeley.zip','size':0,'sha':None}
        else:
            base='https://data.mendeley.com/public-api/datasets/kgbb3yn47p'
            problems=[]
            for url in [base+'?version=3',base+'/files/3',base+'/files?folder_id=root&version=3',base]:
                try:
                    data=self.dl.json(url)
                    if isinstance(data,dict) and 'version' in data and str(data['version'])!='3':
                        raise ValueError('Mendeley 버전 3이 아닌 응답')
                    candidates=[r for r in file_records(data) if r['name'].lower().endswith('.zip')]
                    if not candidates:raise ValueError('ZIP 다운로드 메타데이터 없음')
                    record=min(candidates,key=lambda r:(0 if 'subjectwise' in re.sub(r'[^a-z]','',r['name'].lower()) else 1,r['name']))
                    dump(self.out/'mendeley_provenance.json',{'page':MD_PAGE,'file':record,'license':'CC BY 4.0'})
                    break
                except Exception as e:problems.append(str(e))
            if record is None:raise RuntimeError('공개 API 실패. MENDELEY_ZIP_URL에 공식 ZIP 직접 다운로드 주소를 지정하세요. '+ '; '.join(problems))
        archive=self.dl.get(record['url'],self.work/'mendeley.zip',record['size'],record['sha'])
        before=len(self.records);counts=Counter();discovered=0;unparsed=0
        try:
            with zipfile.ZipFile(archive) as z:
                members=[m for m in z.infolist() if not m.is_dir() and Path(m.filename).suffix.lower() in {'.mp4','.avi','.mov','.mkv','.m4v'} and '__MACOSX' not in m.filename]
                if not members:raise ValueError('ZIP 안에 직접 포함된 영상이 없습니다. 중첩 ZIP은 지원하지 않습니다.')
                discovered=len(members)
                for m in tqdm(members,desc='Mendeley 관절 추출',mininterval=2,file=sys.stdout):
                    try:ex,q,g=parse_mendeley(m.filename)
                    except ValueError as e:
                        unparsed+=1;self.failed.append({'source':'mendeley','name':m.filename,'reason':str(e)});continue
                    limit=self.cfg['MAX_VIDEOS_PER_CLASS']
                    if limit and counts[(ex,q)]>=limit:continue
                    counts[(ex,q)]+=1
                    rec={'source':'mendeley','name':m.filename,'exercise':ex,'quality':q,
                         'group':g,'split':'','errors':{}}
                    temp=self.work/('current'+Path(m.filename).suffix.lower())
                    try:
                        if m.file_size>2*1024**3:raise ValueError('개별 영상이 2 GiB를 초과함')
                        with z.open(m) as src,temp.open('wb') as dst:shutil.copyfileobj(src,dst,2*1024*1024)
                        self.add_video(temp,rec)
                    except PoseEnvironmentError:raise
                    except Exception as e:self.failed.append({**rec,'reason':str(e)})
                    finally:
                        temp.unlink(missing_ok=True)
                        if (len(self.records)+len(self.failed))%25==0:self.progress('pose_extraction')
        finally:archive.unlink(missing_ok=True)
        self.save_failures()
        self.log('mendeley','processed' if len(self.records)>before else 'failed',f'사용 영상 {len(self.records)-before}/{discovered}, 라벨 해석 실패 {unparsed}',
          usable_videos=len(self.records)-before,discovered_videos=discovered)
    def mendeley_local(self,folder):
        root=Path(folder).resolve()
        if not root.is_dir():raise FileNotFoundError(root)
        members=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in {'.mp4','.avi','.mov','.mkv','.m4v'})
        before=len(self.records);counts=Counter()
        for path in tqdm(members,desc='기존 Mendeley 영상 관절 추출',mininterval=2,file=sys.stdout):
            rel=str(path.relative_to(root))
            try:ex,q,g=parse_mendeley(rel)
            except ValueError as e:self.failed.append({'source':'mendeley','name':rel,'reason':str(e)});continue
            limit=self.cfg['MAX_VIDEOS_PER_CLASS']
            if limit and counts[(ex,q)]>=limit:continue
            counts[(ex,q)]+=1
            rec={'source':'mendeley','name':rel,'exercise':ex,'quality':q,'group':g,'split':'','errors':{}}
            temp=self.work/('current'+path.suffix.lower())
            try:
                shutil.copyfile(path,temp);self.add_video(temp,rec)
            except PoseEnvironmentError:raise
            except Exception as e:self.failed.append({**rec,'reason':str(e)})
            finally:temp.unlink(missing_ok=True)
            if (len(self.records)+len(self.failed))%25==0:self.progress('pose_extraction')
        self.save_failures()
        self.log('mendeley','processed' if len(self.records)>before else 'failed',
                 f'기존 영상 {len(self.records)-before}/{len(members)} 사용. 사용자 원본은 삭제하지 않음.',
                 usable_videos=len(self.records)-before,discovered_videos=len(members))
    def ucf101(self):
        # Published archive, not YouTube scraping. Only eight workout categories are used.
        import rarfile
        if not any(shutil.which(x) for x in ['unrar','unar','bsdtar','7z']):
            raise RuntimeError('RAR 해제 도구가 없습니다. apt-get install unrar-free 또는 unar 필요')
        url=self.cfg.get('UCF101_ARCHIVE_URL','').strip() or 'https://www.crcv.ucf.edu/data/UCF101/UCF101.rar'
        arc=self.dl.get(url,self.work/'ucf101.rar')
        before=len(self.records);counts=Counter();discovered=0
        try:
            with rarfile.RarFile(arc) as z:
                members=[]
                for m in z.infolist():
                    match=re.search(r'v_([A-Za-z0-9]+)_g(\d+)_c\d+\.avi$',m.filename)
                    if match and match.group(1) in UCF_MAP:members.append((m,match))
                discovered=len(members)
                for m,match in tqdm(members,desc='UCF 운동 영상 관절 추출',mininterval=2,file=sys.stdout):
                    label=match.group(1);ex=UCF_MAP[label]
                    limit=self.cfg['MAX_VIDEOS_PER_CLASS']
                    if limit and counts[ex]>=limit:continue
                    counts[ex]+=1
                    rec={'source':'ucf101','name':m.filename,'exercise':ex,'quality':'',
                         'group':f'ucf101:{label}:g{match.group(2)}','split':'','errors':{}}
                    temp=self.work/'current.avi'
                    try:
                        with z.open(m) as src,temp.open('wb') as dst:shutil.copyfileobj(src,dst,2*1024*1024)
                        self.add_video(temp,rec)
                    except PoseEnvironmentError:raise
                    except Exception as e:self.failed.append({**rec,'reason':str(e)})
                    finally:
                        temp.unlink(missing_ok=True)
                        if (len(self.records)+len(self.failed))%25==0:self.progress('pose_extraction')
        finally:arc.unlink(missing_ok=True)
        self.log('ucf101','processed',f'운동 8종 중 사용 영상 {len(self.records)-before}/{discovered}. Good/Bad 라벨 없음. 공식 벤치마크 split은 아님.',
                 usable_videos=len(self.records)-before,discovered_videos=discovered,
                 split_policy='original-video group, not guaranteed subject-disjoint')
    def csv_points(self,df,selected):
        p=np.full((len(df),33,4),np.nan,np.float32);p[:,:,3]=0
        for j in selected:
            keys=[NAMES[int(j)]+'_'+s for s in ('x','y','z','v')]
            if not all(k in df for k in keys):raise ValueError('CSV 관절 열 누락: '+str(keys))
            p[:,j]=df[keys].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32)
        return p
    def correction_csvs(self):
        try:
            meta=self.dl.json('https://api.github.com/repos/NgoQuocBao1010/Exercise-Correction/commits/main')
            self.git_revision=meta['sha']
        except Exception:pass
        settings={
          'plank':('plank_model',BODY,{'C':'correct','H':'plank_hips_high','L':'plank_hips_low'}),
          'bicep_curl':('bicep_model',UPPER,{'C':'correct','L':'curl_torso_lean_back'})}
        for ex,(folder,selected,mapping) in settings.items():
            data={};provenance=[]
            for split in ('train','test'):
                url=f'{GH_RAW}{self.git_revision}/core/{folder}/{split}.csv'
                path=self.work/f'{folder}_{split}.csv'
                try:
                    self.dl.get(url,path)
                    df=pd.read_csv(path);df.columns=df.columns.str.strip().str.lower()
                    if 'label' not in df:raise ValueError('label 열 없음')
                    df['label']=df.label.astype(str).str.strip().str.upper()
                    unknown=set(df.label)-set(mapping)
                    if unknown:raise ValueError(f'알 수 없는 오류 라벨: {unknown}')
                    # Remove duplicate coordinate rows; contradictory labels are removed, not guessed.
                    cols=[NAMES[int(j)]+'_'+s for j in selected for s in ('x','y','z','v')]
                    h=pd.util.hash_pandas_object(df[cols],index=False)
                    conflicts=pd.DataFrame({'h':h,'y':df.label}).groupby('h').y.nunique()
                    df=df[~h.isin(conflicts[conflicts>1].index)].drop_duplicates(cols)
                    pts=self.csv_points(df,selected);X,ok=frame_features(pts,selected)
                    y=df.label.map(mapping).to_numpy()[ok];X=X[ok]
                    hashes=pd.util.hash_pandas_object(df.loc[ok,cols],index=False).to_numpy()
                    data[split]=(X,y,hashes,pts[ok])
                    provenance.append({'url':url,'sha256':sha256(path),'rows_used':len(X)})
                except Exception as e:self.log('exercise_correction_'+ex+'_'+split,'failed',e)
                finally:path.unlink(missing_ok=True)
            if 'train' not in data or len(set(data['train'][1]))<2:continue
            X,y,h,pts=data['train'];model=make_model(self.cfg['N_TREES']);model.fit(X,y)
            self.errors[ex]={'model':model,'indices':selected.tolist(),'scope':list(model.classes_),
                             'training_type':'frame','source':GH_REPO,'commit':self.git_revision}
            report={'train_rows':len(y),'classes':dict(Counter(y)),
              'caution':'공개 CSV에 사람/원본 영상 ID가 없어 사람 독립 성능을 검증하지 못함. 랜덤 프레임 분할을 하지 않음.',
              'provenance':provenance}
            if 'test' in data:
                xt,yt,ht,pt=data['test'];keep=~np.isin(ht,h)
                xt,yt,pt=xt[keep],yt[keep],pt[keep]
                report['exact_overlap_removed']=int((~keep).sum())
                report['publisher_test_frame_reference_only']=metrics(yt,model.predict(xt)) if len(yt) else {'n':0}
                data['test']=(xt,yt,ht[keep],pt)
            self.eval['error_'+ex]=report
            if ex=='plank':self.csv_data['plank']=data
            self.log('exercise_correction_'+ex,'trained',f'오류 분류 학습 {len(y)}행. 사람 독립 검증 아님.',classes=list(model.classes_))
    def extra_manifest(self):
        path=self.cfg.get('EXTRA_MANIFEST','').strip()
        if not path:return
        df=pd.read_csv(path,keep_default_na=False,dtype=str)
        required={'source','exercise','subject_id','split','quality','errors_json'}
        if not required.issubset(df):raise ValueError(f'추가 manifest 필수 열: {required}')
        for i,row in df.iterrows():
            r=row.to_dict();source=r['source'];ex=r['exercise'];q=r['quality'].lower().strip()
            if not re.fullmatch(r'[a-z0-9_]+',ex):raise ValueError('운동명은 영문 소문자 snake_case를 사용하세요.')
            if source in {'mendeley','ucf101'}:raise ValueError('추가 source 이름은 기본 데이터와 달라야 합니다.')
            if not source or not r['subject_id']:raise ValueError('source, subject_id/original-video group 필수')
            if q not in {'','good','bad'}:raise ValueError('quality는 good/bad/빈칸만 허용')
            if r['split'] not in {'','train','validation','test'}:raise ValueError('split 값 오류')
            errs=json.loads(r['errors_json']) if r['errors_json'].strip() else {}
            if not isinstance(errs,dict):raise ValueError('errors_json은 JSON 객체여야 합니다.')
            for k,v in errs.items():
                if k not in CORRECTIONS or CORRECTIONS[k]['exercise']!=ex or type(v) is not int or v not in (0,1):
                    raise ValueError(f'확인되지 않은 오류 라벨: {k}={v}; 코드의 CORRECTIONS와 명시적으로 매핑하세요.')
            if q=='good' and any(errs.values()):raise ValueError('Good와 오류=1 라벨이 충돌합니다.')
            rec={'source':source,'name':r.get('video_path') or r.get('video_url') or f'row_{i}',
                 'exercise':ex,'quality':q,'group':source+':'+r['subject_id'],'split':r['split'],'errors':errs}
            temp=self.work/'extra_video.mp4'
            try:
                if r.get('video_path'):shutil.copyfile(r['video_path'],temp)
                elif r.get('video_url'):self.dl.get(r['video_url'],temp)
                else:raise ValueError('video_path 또는 직접 video_url 필수')
                self.add_video(temp,rec)
            except PoseEnvironmentError:raise
            except Exception as e:self.failed.append({**rec,'reason':str(e)})
            finally:temp.unlink(missing_ok=True)
        self.log('extra_manifest','processed',f'{len(df)}개 명시적 라벨 입력 처리; 원본 로컬 파일은 삭제하지 않음')
    def fit_video_models(self):
        if not self.records:
            self.log('main_models','not_trained','유효한 영상 없음. 운동 자동 인식과 다종 Good/Bad는 학습되지 않았습니다.')
            return {'action':None,'action_frame':None,'quality':{},'quality_image':{},'video_errors':{},'error_frame':self.errors}
        split=group_split(self.records)
        for r,s in zip(self.records,split):r['split']=s
        table=pd.DataFrame(self.records);table.to_csv(self.out/'video_manifest_and_splits.csv',index=False,encoding='utf-8-sig')
        bundle={'action':None,'action_frame':None,'quality':{},'quality_image':{},'video_errors':{},'error_frame':self.errors}
        goodvec=np.array([v is not None for v in self.features]);splits=np.array(split)
        exercises=np.array([r['exercise'] for r in self.records]);yq=np.array([r['quality'] for r in self.records])
        # Temporal video-level exercise model.
        tr=np.where(goodvec&(splits=='train'))[0]
        if len(tr)>4 and len(set(exercises[tr]))>=2:
            model=make_model(self.cfg['N_TREES']);model.fit(np.stack([self.features[i] for i in tr]),exercises[tr]);bundle['action']=model
            for part in ['validation','test']:
                ids=np.where(goodvec&(splits==part))[0]
                if len(ids):self.eval['action_'+part]=metrics(exercises[ids],model.predict(np.stack([self.features[i] for i in ids])))
        # Static frame action model adds plank CSV; frames never cross the video split.
        ax=[];ay=[];aw=[]
        for i,r in enumerate(self.records):
            if r['split']!='train':continue
            ax.append(self.frames[i]);ay.extend([r['exercise']]*len(self.frames[i]));aw.extend([1/len(self.frames[i])]*len(self.frames[i]))
        if 'plank' in self.csv_data:
            pts=self.csv_data['plank']['train'][3];px,ok=frame_features(pts);px=px[ok]
            if len(px):
                # One big frame CSV must not dominate the number of independent videos.
                choose=np.linspace(0,len(px)-1,min(3000,len(px))).astype(int);px=px[choose]
                ax.append(px);ay.extend(['plank']*len(px));aw.extend([min(50,max(5,len(self.records)/8))/len(px)]*len(px))
        if len(set(ay))>=2:
            fm=make_model(self.cfg['N_TREES']);fm.set_params(class_weight=None)
            ay=np.array(ay);aw=np.array(aw,dtype=float)
            for label in set(ay):
                mask=ay==label;aw[mask]/=max(aw[mask].sum(),1e-12)
            fm.fit(np.concatenate(ax),ay,sample_weight=aw);bundle['action_frame']=fm
            ys=[];ps=[]
            for i,r in enumerate(self.records):
                if r['split']=='test':
                    ys.append(r['exercise']);ps.append(str(fm.classes_[fm.predict_proba(self.frames[i]).mean(0).argmax()]))
            self.eval['frame_action_video_test']=metrics(ys,ps)
            self.eval['plank_action_warning']='플랭크 자료의 사람/영상 ID 없음. 플랭크 일반화 성능은 별도 검증 필요.'
        self.checkpoint(bundle,'action_classifiers_fitted')
        # Quality is learned ONLY from known quality labels. Missing is not Good.
        for ex in sorted(set(exercises)):
            print('Good/Bad 학습 대상:',EX_KO.get(ex,ex),flush=True)
            ids=np.where(goodvec&(splits=='train')&(exercises==ex)&np.isin(yq,['good','bad']))[0]
            groups={self.records[i]['group'] for i in ids}
            if len(ids)<8 or len(groups)<2 or set(yq[ids])!={'good','bad'}:continue
            model=make_model(self.cfg['N_TREES']);model.fit(np.stack([self.features[i] for i in ids]),yq[ids]);bundle['quality'][ex]=model
            for part in ['validation','test']:
                ix=np.where(goodvec&(splits==part)&(exercises==ex)&np.isin(yq,['good','bad']))[0]
                report=metrics(yq[ix],model.predict(np.stack([self.features[i] for i in ix]))) if len(ix) else {'n':0}
                report['train_videos']=len(ids);report['train_groups']=len(groups)
                self.eval[f'quality_{ex}_{part}']=report
        # Fallback is also video-level and group-disjoint. Image coordinates are not physical 3D angles.
        for ex in sorted(set(exercises)):
            ids=np.where((splits=='train')&(exercises==ex)&np.isin(yq,['good','bad']))[0]
            groups={self.records[i]['group'] for i in ids}
            if len(ids)<8 or len(groups)<2 or set(yq[ids])!={'good','bad'}:continue
            model=make_model(self.cfg['N_TREES'])
            model.fit(np.stack([self.summaries[i] for i in ids]),yq[ids])
            bundle['quality_image'][ex]=model
            for part in ['validation','test']:
                ix=np.where((splits==part)&(exercises==ex)&np.isin(yq,['good','bad']))[0]
                report=metrics(yq[ix],model.predict(np.stack([self.summaries[i] for i in ix]))) if len(ix) else {'n':0}
                report['train_videos']=len(ids);report['train_groups']=len(groups)
                self.eval[f'quality_image_{ex}_{part}']=report
        self.checkpoint(bundle,'core_classifiers_fitted')
        # Optional, explicitly annotated video error models. Unknown error labels are masked out.
        for code,info in CORRECTIONS.items():
            ex=info['exercise']
            ids=[i for i,r in enumerate(self.records) if goodvec[i] and split[i]=='train' and r['exercise']==ex and code in r['errors']]
            target=np.array([self.records[i]['errors'][code] for i in ids])
            if len(ids)<8 or len(set(target))<2 or len({self.records[i]['group'] for i in ids})<2:continue
            model=make_model(self.cfg['N_TREES']);model.fit(np.stack([self.features[i] for i in ids]),target)
            bundle['video_errors'][code]=model
            ix=[i for i,r in enumerate(self.records) if goodvec[i] and split[i]=='test' and r['exercise']==ex and code in r['errors']]
            if ix:self.eval['video_error_'+code]=metrics([self.records[i]['errors'][code] for i in ix],model.predict(np.stack([self.features[i] for i in ix])))
        rows=[]
        for i,r in enumerate(self.records):
            if r['split']!='test':continue
            vec=self.features[i];scores={};pred=''
            if bundle['action'] is not None and vec is not None:
                probs=bundle['action'].predict_proba(vec[None])[0]
                pred=str(bundle['action'].classes_[int(probs.argmax())]);score=float(probs.max())
            elif bundle['action_frame'] is not None:
                probs=bundle['action_frame'].predict_proba(self.frames[i]).mean(0)
                pred=str(bundle['action_frame'].classes_[int(probs.argmax())]);score=float(probs.max())
            else:score=None
            qpred='';qscore=None;mode=''
            if pred in bundle['quality'] and vec is not None:
                m=bundle['quality'][pred];probs=m.predict_proba(vec[None])[0];mode='temporal'
            elif pred in bundle['quality_image']:
                m=bundle['quality_image'][pred];probs=m.predict_proba(self.summaries[i][None])[0];mode='image_summary'
            else:m=None
            if m is not None:qpred=str(m.classes_[int(probs.argmax())]);qscore=float(probs.max())
            rows.append({'source':r['source'],'name':r['name'],'group':r['group'],
                'true_exercise':r['exercise'],'predicted_exercise':pred,'action_score':score,
                'true_quality':r['quality'],'predicted_quality':qpred,'quality_score':qscore,'quality_features':mode})
        pd.DataFrame(rows).to_csv(self.out/'test_predictions.csv',index=False,encoding='utf-8-sig')
        labelled=[r for r in rows if r['true_quality'] in {'good','bad'}]
        covered=[r for r in labelled if r['predicted_quality']]
        self.eval['cascade_test_reference']={'n_labelled':len(labelled),'n_quality_predicted':len(covered),
            'coverage':len(covered)/len(labelled) if labelled else None,
            'joint_accuracy_counting_missing_as_wrong':sum(r['true_exercise']==r['predicted_exercise'] and r['true_quality']==r['predicted_quality'] for r in labelled)/len(labelled) if labelled else None,
            'note':'원시 cascade argmax 참고 평가. 배포 predict의 점수 임계값/충돌 보류 정책 전체를 적용한 성능은 아님.'}
        return bundle
    def run(self):
        self.dl.get(MODEL_URL,self.asset)
        print('관절 추출 환경 사전 검사 중 (아직 운동 데이터는 받지 않았습니다)...',flush=True)
        try:
            preflight=pose_preflight(self.asset,self.work,self.cfg)
            dump(self.out/'preflight.json',preflight)
            self.log('preflight','passed','Agg backend + 영상 디코딩 + MediaPipe 실제 추론 확인')
        except Exception as e:
            dump(self.out/'preflight.json',{'status':'failed','reason':str(e),'traceback':traceback.format_exc()})
            self.log('preflight','failed',e)
            raise
        # Fast, genuinely labeled correction data first; raw repository models are NEVER loaded.
        if self.cfg['USE_CORRECTION_CSV']:
            self.correction_csvs()
            self.checkpoint({'action':None,'action_frame':None,'quality':{},'quality_image':{},'video_errors':{},'error_frame':self.errors},'correction_classifiers_fitted')
        if self.cfg['USE_MENDELEY']:
            try:
                if self.cfg.get('EXISTING_VIDEO_DIR'):self.mendeley_local(self.cfg['EXISTING_VIDEO_DIR'])
                else:self.mendeley()
            except PoseEnvironmentError:raise
            except Exception as e:self.log('mendeley','failed',e)
        if self.cfg['USE_UCF101']:
            try:self.ucf101()
            except PoseEnvironmentError:raise
            except Exception as e:self.log('ucf101','failed',e)
        else:
            self.log('ucf101','disabled','기본 실행에서 제외: 이전 HTTPS 인증서 오류. Mendeley + 오류 CSV만 사용. SSL 검증을 끄지 않습니다.')
        try:self.extra_manifest()
        except Exception as e:
            self.log('extra_manifest','failed',e)
            # Invalid explicit annotations must not be partially accepted unnoticed.
            raise
        for source,reason in [
          ('Fitness-AQA','자동 포함 안 됨: 저자 승인/비상업 조건. 승인받은 영상과 라벨을 EXTRA_MANIFEST로 입력.'),
          ('Kinetics','자동 전체 수집 안 됨: 대형 행동 데이터이며 Good/Bad 라벨 없음. 권한 있는 영상 직접 URL을 manifest로 입력.'),
          ('InfiniteRep','자동 포함 안 됨: 원본 배포 접근과 라벨을 확인해야 함. 확보한 파일을 manifest로 입력. 합성 영상을 자동 Good로 처리하지 않음.'),
          ('Zenodo Squat 17558630','자동 포함 안 됨: 정지 이미지 데이터입니다. 이 영상 학습 입력기에 이미지들을 영상인 것처럼 넣지 않습니다.'),
          ('Kaggle Workout','자동 포함 안 됨: 게시한 글에 특정 데이터셋 ID가 없음. 확보한 자료를 manifest로 입력.'),
          ('PosePilot / fitness-trainer-pose-estimation','저장소 코드를 데이터로 간주하거나 기존 모델을 재학습했다고 표시하지 않음.')]:
            self.log(source,'not_auto_included',reason)
        self.save_failures()
        print('운동 인식 / Good-Bad 분류기 학습 시작. 유효 영상:',len(self.records),flush=True)
        bundle=self.fit_video_models()
        bundle.update({'feature_version':FEATURE_VERSION,'config':{k:v for k,v in self.cfg.items() if k not in {'WORK','EXTRA_MANIFEST'}},
          'corrections':CORRECTIONS,'created_at':time.strftime('%Y-%m-%dT%H:%M:%S'),
          'score_note':'분류기 점수는 보정된 정확도/안전 확률이 아닙니다.'})
        atomic_model_save(bundle,self.out/'fitness_models.joblib')
        dump(self.out/'metrics.json',self.eval)
        support={'action_temporal':list(bundle['action'].classes_) if bundle['action'] is not None else [],
          'action_frame':list(bundle['action_frame'].classes_) if bundle['action_frame'] is not None else [],
          'quality_full_clip':sorted(set(bundle['quality'])|set(bundle['quality_image'])),
          'quality_temporal':sorted(bundle['quality']),
          'quality_image_summary_fallback':sorted(bundle['quality_image']),
          'limited_error_checks':{ex:v['scope'] for ex,v in self.errors.items()},
          'video_error_models':sorted(bundle['video_errors']),
          'usable_video_counts':dict(Counter(r['source'] for r in self.records)),
          'failed_or_skipped_video_count':len(self.failed),
          'downloaded_GiB':self.dl.used/1024**3,
          'main_pipeline_trained':(bundle['action'] is not None or bundle['action_frame'] is not None) and bool(bundle['quality'] or bundle['quality_image']),
          'mendeley_all_7_quality_trained':set(PATTERNS).issubset(set(bundle['quality'])|set(bundle['quality_image'])),
          'missing_mendeley_quality_models':sorted(set(PATTERNS)-(set(bundle['quality'])|set(bundle['quality_image']))),
          'correction_models_trained':{'plank','bicep_curl'}.issubset(self.errors) and {'correct','plank_hips_high','plank_hips_low'}.issubset(self.errors.get('plank',{}).get('scope',[])) and {'correct','curl_torso_lean_back'}.issubset(self.errors.get('bicep_curl',{}).get('scope',[])),
          'feature_version':FEATURE_VERSION,
          'warning':'공개 데이터 전체를 학습한 것이 아님. source_status.json 확인. 임상/안전 판정 아님.'}
        dump(self.out/'capabilities.json',support)
        print('\n=== 실제 학습 완료 범위 ===\n'+json.dumps(support,ensure_ascii=False,indent=2),flush=True)
        if not support['main_pipeline_trained']:
            self.log('main_models','incomplete','운동 인식 + Good/Bad 핵심 모델 미완성. 부분 모델과 실패 보고서만 저장했습니다.')
            raise RuntimeError('핵심 영상 학습 실패. CSV 일부 학습을 전체 완료로 처리하지 않습니다. failed_or_skipped_videos.csv를 확인하세요.')
        if self.cfg.get('REQUIRE_ALL_MENDELEY_7',True) and self.cfg.get('USE_MENDELEY') and not support['mendeley_all_7_quality_trained']:
            raise RuntimeError('7종 전체 학습 미완성. 부분 모델은 저장했으나 성공으로 처리하지 않습니다. 누락: '+str(support['missing_mendeley_quality_models']))
        if self.cfg.get('USE_CORRECTION_CSV') and not support['correction_models_trained']:
            raise RuntimeError('요청한 플랭크/컬 오류 분류가 모두 학습되지 않았습니다. 부분 모델은 저장했습니다. source_status.json을 확인하세요.')
        if self.cfg.get('TEST_VIDEO_PATH'):
            try:
                result=predict_video(self.out,self.cfg['TEST_VIDEO_PATH'],self.cfg.get('EXERCISE_HINT',''))
                dump(self.out/'prediction.json',result);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
            except Exception as e:
                dump(self.out/'prediction_failure.json',{'reason':str(e)})
                self.log('test_prediction','failed',e)
        return support


def predict_video(folder,video,exercise_hint=''):
    folder=Path(folder)
    # joblib uses pickle: only load an artifact that YOU produced and trust.
    b=joblib.load(folder/'fitness_models.joblib');cfg=b['config']
    try:points,world,t=extract_pose(video,folder/'pose_landmarker_full.task',cfg)
    except Exception as e:return {'exercise':'Unknown','quality':'판단 보류','reason':str(e),'correction':[]}
    ff,valid=frame_features(points)
    if valid.sum()<12 or valid.mean()<.55:
        return {'exercise':'Unknown','quality':'판단 보류','reason':'가림/촬영 조건으로 관절 검출 부족','correction':[]}
    try:vec=temporal_features(world,points[:,:,3],t)
    except ValueError:vec=None
    action_scores={};static_scores={};ex=exercise_hint.strip();conf=None
    if not ex:
        if b['action'] is not None and vec is not None:
            p=b['action'].predict_proba(vec[None])[0]
            action_scores=dict(zip(b['action'].classes_,map(float,p)))
        if b['action_frame'] is not None:
            p=b['action_frame'].predict_proba(ff[valid]).mean(0)
            static_scores=dict(zip(b['action_frame'].classes_,map(float,p)))
        # Static fallback adds plank, but contradictory action heads result in abstention.
        if action_scores:
            ex=max(action_scores,key=action_scores.get);conf=action_scores[ex]
            if static_scores:
                s=max(static_scores,key=static_scores.get)
                if s=='plank' and static_scores[s]>=.85:
                    if conf>=.8 and ex!='plank':
                        return {'exercise':'Unknown','quality':'판단 보류','reason':'동작 모델과 플랭크 모델이 충돌함. EXERCISE_HINT로 운동을 지정해 재검사하세요.','correction':[]}
                    ex,conf=s,static_scores[s]
                elif static_scores.get(s,0)>=.85 and s!=ex and conf>=.7:
                    return {'exercise':'Unknown','quality':'판단 보류','reason':'운동 인식 모델 간 불일치','correction':[]}
        elif static_scores:ex=max(static_scores,key=static_scores.get);conf=static_scores[ex]
        else:return {'exercise':'Unknown','quality':'판단 보류','reason':'운동 인식 모델이 없음. 지원 운동의 EXERCISE_HINT가 필요함.','correction':[]}
        if conf<.60:return {'exercise':'Unknown','quality':'판단 보류','reason':'운동 인식 점수 낮음','action_scores':action_scores or static_scores,'correction':[]}
    result={'exercise':str(ex),'exercise_ko':EX_KO.get(ex,ex),'exercise_source':'user_hint' if exercise_hint else 'model',
      'action_score':conf,'analyzed_frame_count':int(len(t)),
      'sample_interval_seconds':float(np.median(np.diff(t))),
      'sampling_note':'일부 프레임만 분석하므로 짧은 오류를 놓칠 수 있음. 긴 영상은 한 운동의 짧은 구간으로 잘라 사용하세요.',
      'quality':'판단 보류','quality_scope':'unknown','quality_score':None,
      'suspected_errors':[],'correction':[],
      'notice':'연구용 분류 결과입니다. 점수는 안전 확률이 아니며, 통증이 있으면 동작을 멈추고 전문가에게 확인하세요.'}
    generic_quality=None
    if ex in b['quality'] and vec is not None:
        m=b['quality'][ex];p=m.predict_proba(vec[None])[0];j=int(p.argmax())
        result['quality_score']=float(p[j]);result['quality_scope']='dataset_clip_good_bad'
        if p[j]>=.70:generic_quality=str(m.classes_[j]);result['quality']=generic_quality.title()
    if ex in b.get('quality_image',{}):
        m=b['quality_image'][ex];ip=m.predict_proba(image_summary(ff[valid])[None])[0]
        ij=int(ip.argmax());iq=str(m.classes_[ij]);isc=float(ip[ij])
        result['image_summary_quality_score']=isc
        if generic_quality is None and isc>=.75:
            generic_quality=iq;result['quality']=iq.title()
            result['quality_score']=isc;result['quality_scope']='dataset_clip_good_bad_image_summary'
        elif generic_quality is not None and iq!=generic_quality and isc>=.85:
            result['quality']='판단 보류';result['reason']='시간 특징 모델과 영상 좌표 요약 모델이 충돌함'
            result['quality_model_conflict']=True
    checks=[];limited_correct=False
    if ex in b['error_frame']:
        spec=b['error_frame'][ex];xf,ok=frame_features(points,np.array(spec['indices']))
        if ok.sum()>=12 and ok.mean()>=.55:
            p=spec['model'].predict_proba(xf[ok]);classes=list(spec['model'].classes_)
            seconds=t[ok];original_indices=np.flatnonzero(ok)
            for j,code in enumerate(classes):
                strong=p[:,j]>=.80
                # Require persistence, not one uncertain frame. These are heuristic decision thresholds.
                proportion=float(strong.mean())
                if code=='correct':limited_correct=proportion>=.80;continue
                hits=np.where(strong)[0];longest=0;start=last=None
                for k in hits:
                    if last is None or k!=last+1 or original_indices[k]!=original_indices[last]+1:start=k
                    longest=max(longest,float(seconds[k]-seconds[start]));last=k
                if proportion>=.25 and longest>=.5:
                    checks.append({'code':str(code),'score_mean':float(p[:,j].mean()),
                      'strong_frame_fraction':proportion,'longest_span_seconds':longest,
                      'evidence':'frame_classifier_experimental'})
    for code,m in b['video_errors'].items():
        if CORRECTIONS[code]['exercise']!=ex or vec is None:continue
        p=m.predict_proba(vec[None])[0];j=list(m.classes_).index(1)
        if p[j]>=.80:checks.append({'code':code,'score':float(p[j]),'evidence':'explicitly_annotated_video_classifier'})
    result['suspected_errors']=checks
    if checks:
        if generic_quality=='good' or result.get('quality_model_conflict'):
            result['quality']='판단 보류';result['reason']='Good/Bad 모델과 오류 모델의 결과가 충돌함'
        else:
            result['quality']='Bad';result['quality_scope']='at_least_one_trained_error_suspected'
        for item in checks:
            c=CORRECTIONS[item['code']]
            result['correction'].append({'suspected_issue':c['name'],'suggestion':c['tip'],
                                         'type':'고정된 일반 코칭 문구; 의학적 진단 아님'})
    elif ex=='plank' and limited_correct and generic_quality is None:
        result['quality']='Good';result['quality_scope']='플랭크의 골반 높음/처짐 두 오류에 한정; 전체 안전 보장 아님'
    if result['quality']=='Bad' and not checks:
        result['reason']='Good/Bad 라벨만 학습한 운동이어서 구체적인 오류 원인은 학습되지 않음'
        result['correction']=[{'type':'원인 진단이 아닌 일반 점검 목록','suggestion':CHECKLIST.get(ex,'정면과 측면에서 다시 촬영해 숙련된 지도자와 자세를 확인하세요.')}]
    if result['quality']=='판단 보류' and 'reason' not in result:
        result['reason']='자세 라벨/모델 부족, 관절 가림, 또는 판별 점수 부족. 운동 인식만으로 Good를 부여하지 않음.'
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config');parser.add_argument('--predict');parser.add_argument('--bundle');parser.add_argument('--exercise',default='');parser.add_argument('--output',default='')
    args=parser.parse_args()
    if args.predict:
        if not args.bundle:parser.error('--bundle 필요')
        result=predict_video(args.bundle,args.predict,args.exercise)
        if args.output:dump(args.output,result)
        print(json.dumps(result,ensure_ascii=False,indent=2));return
    if not args.config:parser.error('--config 필요')
    cfg=json.loads(Path(args.config).read_text());pipe=Pipeline(cfg)
    try:pipe.run()
    except BaseException:
        pipe.save_failures()
        dump(pipe.out/'failure.json',{'error':traceback.format_exc()})
        raise
    finally:
        # Delete only the unique temp directory created by the wrapper, never user data or Drive.
        work=Path(cfg['WORK']).resolve()
        if cfg.get('DELETE_TEMP_DATA',True) and work.parent==Path('/content') and work.name.startswith('fitness_temp_'):
            shutil.rmtree(work,ignore_errors=True)
            print('직접 생성한 임시 다운로드/영상/CSV를 삭제했습니다. 모델·보고서는 유지합니다.',flush=True)

if __name__=='__main__':main()


