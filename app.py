
#!/usr/bin/env python3
'''
Voice Emotion Diagnosis Web App - Standalone
Universidad Panamericana - Deep Learning
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf
import json
import os
import gradio as gr
import warnings
import math

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SR = 22050
DURATION = 3.0
N_MFCC = 40
HOP = 512
N_FFT = 2048

with open(os.path.join(BASE_DIR, 'config.json')) as f:
    CONFIG = json.load(f)

EMOTIONS = CONFIG['emotions']
EMOJIS = CONFIG['emojis']
NUM_CLASSES = len(EMOTIONS)
CHAMPION_NAME = CONFIG['champion_name']
CHAMP_INPUT = CONFIG['champion_input_type']

with open(os.path.join(BASE_DIR, 'scalers.json')) as f:
    SCALERS = json.load(f)
SCALER_MEAN = np.array(SCALERS['classic_mean'], dtype=np.float32)
SCALER_SCALE = np.array(SCALERS['classic_scale'], dtype=np.float32)
SCALER_EMB_MEAN = np.array(SCALERS['emb_mean'], dtype=np.float32)
SCALER_EMB_SCALE = np.array(SCALERS['emb_scale'], dtype=np.float32)

with open(os.path.join(BASE_DIR, 'feature_cols.json')) as f:
    FEATURE_COLS = json.load(f)

N_FEAT = CONFIG['n_features']
N_EMB = CONFIG['n_emb']

# --- ARCHITECTURES ---
class FFNN(nn.Module):
    def __init__(self, d, n=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,512),nn.BatchNorm1d(512),nn.GELU(),nn.Dropout(0.3),nn.Linear(512,256),nn.BatchNorm1d(256),nn.GELU(),nn.Dropout(0.3),nn.Linear(256,128),nn.BatchNorm1d(128),nn.GELU(),nn.Dropout(0.2),nn.Linear(128,64),nn.BatchNorm1d(64),nn.GELU(),nn.Dropout(0.2),nn.Linear(64,n))
    def forward(self,x): return self.net(x)

class CNN1D(nn.Module):
    def __init__(self, d, n=8):
        super().__init__()
        self.proj=nn.Linear(1,32)
        self.conv=nn.Sequential(nn.Conv1d(32,64,3,padding=1),nn.BatchNorm1d(64),nn.GELU(),nn.MaxPool1d(2),nn.Dropout(0.2),nn.Conv1d(64,128,3,padding=1),nn.BatchNorm1d(128),nn.GELU(),nn.MaxPool1d(2),nn.Dropout(0.2),nn.Conv1d(128,256,3,padding=1),nn.BatchNorm1d(256),nn.GELU(),nn.AdaptiveAvgPool1d(4))
        self.head=nn.Sequential(nn.Linear(1024,128),nn.GELU(),nn.Dropout(0.3),nn.Linear(128,n))
    def forward(self,x): x=self.proj(x.unsqueeze(-1)).permute(0,2,1); return self.head(self.conv(x).flatten(1))

class ResBlock(nn.Module):
    def __init__(self,d,dr=0.2):
        super().__init__()
        self.block=nn.Sequential(nn.BatchNorm1d(d),nn.GELU(),nn.Linear(d,d),nn.BatchNorm1d(d),nn.GELU(),nn.Dropout(dr),nn.Linear(d,d))
        self.drop=nn.Dropout(dr)
    def forward(self,x): return self.drop(x+self.block(x))

class ResNet(nn.Module):
    def __init__(self,d,n=8,h=256,nb=4):
        super().__init__()
        self.proj=nn.Sequential(nn.Linear(d,h),nn.BatchNorm1d(h),nn.GELU())
        self.blocks=nn.Sequential(*[ResBlock(h) for _ in range(nb)])
        self.head=nn.Sequential(nn.BatchNorm1d(h),nn.Linear(h,64),nn.GELU(),nn.Dropout(0.3),nn.Linear(64,n))
    def forward(self,x): return self.head(self.blocks(self.proj(x)))

class LSTMModel(nn.Module):
    def __init__(self,d,n=8,h=128,nl=2,t=8):
        super().__init__()
        self.t=t; self.nt=(d+t-1)//t; self.pd=self.nt*t
        self.proj=nn.Linear(t,64); self.lstm=nn.LSTM(64,h,nl,batch_first=True,dropout=0.3)
        self.head=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,64),nn.GELU(),nn.Dropout(0.3),nn.Linear(64,n))
    def forward(self,x):
        B=x.shape[0]
        if x.shape[1]<self.pd: x=F.pad(x,(0,self.pd-x.shape[1]))
        x=self.proj(x.view(B,self.nt,self.t)); _,(hn,_)=self.lstm(x); return self.head(hn[-1])

class BahdanauAttention(nn.Module):
    def __init__(self,h): super().__init__(); self.W=nn.Linear(h,h); self.V=nn.Linear(h,1)
    def forward(self,s): w=F.softmax(self.V(torch.tanh(self.W(s))),dim=1); return torch.sum(w*s,dim=1),w.squeeze(-1)

class BiLSTMAttention(nn.Module):
    def __init__(self,d,n=8,h=128,nl=2,t=8):
        super().__init__()
        self.t=t; self.nt=(d+t-1)//t; self.pd=self.nt*t
        self.proj=nn.Linear(t,64); self.bilstm=nn.LSTM(64,h,nl,batch_first=True,dropout=0.3,bidirectional=True)
        self.attn=BahdanauAttention(h*2)
        self.head=nn.Sequential(nn.LayerNorm(h*2),nn.Linear(h*2,128),nn.GELU(),nn.Dropout(0.3),nn.Linear(128,64),nn.GELU(),nn.Dropout(0.2),nn.Linear(64,n))
    def forward(self,x):
        B=x.shape[0]
        if x.shape[1]<self.pd: x=F.pad(x,(0,self.pd-x.shape[1]))
        x=self.proj(x.view(B,self.nt,self.t)); o,_=self.bilstm(x); c,_=self.attn(o); return self.head(c)

class TransformerModel(nn.Module):
    def __init__(self,d,n=8,dm=128,nh=8,nl=4):
        super().__init__()
        self.t=8; self.nt=(d+self.t-1)//self.t; self.pd=self.nt*self.t
        self.proj=nn.Linear(self.t,dm); self.cls=nn.Parameter(torch.randn(1,1,dm)*0.02)
        pe=torch.zeros(self.nt+1,dm); pos=torch.arange(self.nt+1).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,dm,2).float()*-(math.log(10000)/dm))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div[:dm//2])
        self.register_buffer('pe',pe.unsqueeze(0))
        l=nn.TransformerEncoderLayer(dm,nh,dm*4,0.3,'gelu',batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(l,nl); self.norm=nn.LayerNorm(dm)
        self.head=nn.Sequential(nn.Linear(dm,64),nn.GELU(),nn.Dropout(0.3),nn.Linear(64,n))
    def forward(self,x):
        B=x.shape[0]
        if x.shape[1]<self.pd: x=F.pad(x,(0,self.pd-x.shape[1]))
        x=self.proj(x.view(B,self.nt,self.t)); x=torch.cat([self.cls.expand(B,-1,-1),x],1)+self.pe[:,:self.nt+1]
        return self.head(self.norm(self.transformer(x))[:,0])

class ConvTransformer(nn.Module):
    def __init__(self,d,n=8,dm=128,nh=4,nl=3):
        super().__init__()
        self.proj=nn.Linear(1,32)
        self.conv=nn.Sequential(nn.Conv1d(32,64,5,padding=2),nn.BatchNorm1d(64),nn.GELU(),nn.Conv1d(64,128,3,padding=1),nn.BatchNorm1d(128),nn.GELU(),nn.MaxPool1d(2),nn.Dropout(0.15),nn.Conv1d(128,128,3,padding=1),nn.BatchNorm1d(128),nn.GELU(),nn.MaxPool1d(2),nn.Dropout(0.15))
        self.cproj=nn.Linear(128,dm); self.cls=nn.Parameter(torch.randn(1,1,dm)*0.02)
        l=nn.TransformerEncoderLayer(dm,nh,dm*4,0.3,'gelu',batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(l,nl); self.norm=nn.LayerNorm(dm)
        self.head=nn.Sequential(nn.Linear(dm,64),nn.GELU(),nn.Dropout(0.3),nn.Linear(64,n))
    def forward(self,x):
        B=x.shape[0]; x=self.proj(x.unsqueeze(-1)).permute(0,2,1); x=self.conv(x).permute(0,2,1); x=self.cproj(x)
        x=torch.cat([self.cls.expand(B,-1,-1),x],1); return self.head(self.norm(self.transformer(x))[:,0])

class FineTunedHead(nn.Module):
    def __init__(self,d=768,n=8):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,512),nn.LayerNorm(512),nn.GELU(),nn.Dropout(0.3),nn.Linear(512,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(0.3),nn.Linear(256,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(0.2),nn.Linear(128,n))
    def forward(self,x): return self.net(x)

class StudentNet(nn.Module):
    def __init__(self,d,n=8):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,128),nn.BatchNorm1d(128),nn.GELU(),nn.Dropout(0.2),nn.Linear(128,64),nn.BatchNorm1d(64),nn.GELU(),nn.Dropout(0.2),nn.Linear(64,n))
    def forward(self,x): return self.net(x)

# --- LOAD MODEL ---
MODEL_MAP = {
    'Feed-Forward NN': (FFNN, {'d': N_FEAT}), '1D-CNN': (CNN1D, {'d': N_FEAT}),
    'ResNet': (ResNet, {'d': N_FEAT}), 'LSTM': (LSTMModel, {'d': N_FEAT}),
    'BiLSTM+Attention': (BiLSTMAttention, {'d': N_FEAT}),
    'Transformer': (TransformerModel, {'d': N_FEAT}), 'Conv-Transformer': (ConvTransformer, {'d': N_FEAT}),
    'Fine-Tuned (HuBERT)': (FineTunedHead, {'d': N_EMB}),
    'Distilled Student': (StudentNet, {'d': N_EMB if CHAMP_INPUT == 'emb' else N_FEAT}),
}
cls, kw = MODEL_MAP[CHAMPION_NAME]
model = cls(**kw)
model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'model.pt'), map_location=DEVICE, weights_only=False))
model.to(DEVICE).eval()
print(f"Loaded: {CHAMPION_NAME} ({sum(p.numel() for p in model.parameters()):,} params)")

INF_PROC = INF_BACKBONE = None
if CHAMP_INPUT == 'emb':
    try:
        from transformers import AutoProcessor, AutoModel
        INF_PROC = AutoProcessor.from_pretrained("facebook/hubert-base-ls960")
        INF_BACKBONE = AutoModel.from_pretrained("facebook/hubert-base-ls960").to(DEVICE).eval()
        print("HuBERT loaded")
    except:
        try:
            INF_PROC = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
            INF_BACKBONE = AutoModel.from_pretrained("facebook/wav2vec2-base").to(DEVICE).eval()
        except:
            print("WARNING: no pretrained backbone")

# --- FEATURES ---
def extract_features(path):
    try:
        y, _ = librosa.load(path, sr=SR, duration=DURATION)
        t = int(SR * DURATION)
        y = np.pad(y, (0, max(0, t - len(y))))[:t]
        f = {}
        mfccs = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
        for i in range(N_MFCC): f[f'mfcc_{i}_m']=np.mean(mfccs[i]); f[f'mfcc_{i}_s']=np.std(mfccs[i])
        d1=librosa.feature.delta(mfccs); d2=librosa.feature.delta(mfccs,order=2)
        for i in range(13): f[f'd_{i}']=np.mean(d1[i]); f[f'd2_{i}']=np.mean(d2[i])
        mel=librosa.power_to_db(librosa.feature.melspectrogram(y=y,sr=SR,n_mels=128,hop_length=HOP))
        f['mel_m']=np.mean(mel); f['mel_s']=np.std(mel); f['mel_x']=np.max(mel)
        sc=librosa.feature.spectral_centroid(y=y,sr=SR,hop_length=HOP)[0]
        f['sc_m']=np.mean(sc); f['sc_s']=np.std(sc)
        sb=librosa.feature.spectral_bandwidth(y=y,sr=SR,hop_length=HOP)[0]
        f['sb_m']=np.mean(sb); f['sb_s']=np.std(sb)
        f['sr_m']=np.mean(librosa.feature.spectral_rolloff(y=y,sr=SR,hop_length=HOP)[0])
        scon=librosa.feature.spectral_contrast(y=y,sr=SR,hop_length=HOP)
        for i in range(scon.shape[0]): f[f'scon_{i}']=np.mean(scon[i])
        f['sf_m']=np.mean(librosa.feature.spectral_flatness(y=y,hop_length=HOP)[0])
        ch=librosa.feature.chroma_stft(y=y,sr=SR,hop_length=HOP)
        for i in range(12): f[f'chr_{i}']=np.mean(ch[i])
        tn=librosa.feature.tonnetz(y=librosa.effects.harmonic(y),sr=SR)
        for i in range(6): f[f'tn_{i}']=np.mean(tn[i])
        rms=librosa.feature.rms(y=y,hop_length=HOP)[0]; f['rms_m']=np.mean(rms); f['rms_s']=np.std(rms)
        zcr=librosa.feature.zero_crossing_rate(y=y,hop_length=HOP)[0]; f['zcr_m']=np.mean(zcr); f['zcr_s']=np.std(zcr)
        pitches,mags=librosa.piptrack(y=y,sr=SR,n_fft=N_FFT,hop_length=HOP)
        pv=[pitches[mags[:,t].argmax(),t] for t in range(pitches.shape[1])]; pv=[p for p in pv if p>0]
        f['p_m']=np.mean(pv) if pv else 0; f['p_s']=np.std(pv) if pv else 0; f['p_r']=np.ptp(pv) if pv else 0
        tempo,_=librosa.beat.beat_track(y=y,sr=SR); f['tempo']=float(tempo) if np.isscalar(tempo) else float(tempo[0])
        h=librosa.effects.harmonic(y); p=librosa.effects.percussive(y)
        f['hnr']=10*np.log10((np.sum(h**2)+1e-8)/(np.sum(p**2)+1e-8))
        if len(pv)>2: per=1.0/(np.array(pv)+1e-8); f['jitter']=np.mean(np.abs(np.diff(per)))/(np.mean(per)+1e-8)
        else: f['jitter']=0
        f['shimmer']=np.mean(np.abs(np.diff(rms)))/(np.mean(rms)+1e-8) if len(rms)>2 else 0
        return f
    except: return None

# --- DIAGNOSIS ---
POSITIVE = {'happy','surprised','calm'}
NEGATIVE = {'sad','angry','fearful','disgust'}

def diagnosis(pd):
    se = sorted(pd.items(), key=lambda x: x[1], reverse=True)
    dom=se[0]; sec=se[1] if len(se)>1 else None
    pos=sum(pd.get(e,0) for e in POSITIVE); neg=sum(pd.get(e,0) for e in NEGATIVE); neu=pd.get('neutral',0)
    L=["="*50, "  DIAGNOSTICO EMOCIONAL POR VOZ", "="*50]
    L.append(f"\nEmocion dominante: {EMOJIS.get(dom[0],'')} {dom[0].upper()} ({dom[1]*100:.1f}%)")
    if sec and sec[1]>0.12: L.append(f"Emocion secundaria: {EMOJIS.get(sec[0],'')} {sec[0]} ({sec[1]*100:.1f}%)")
    L += [f"\n{'-'*50}","Balance emocional:",f"  Positivas: {pos*100:.1f}%",f"  Negativas: {neg*100:.1f}%",f"  Neutral: {neu*100:.1f}%"]
    L.append(f"\n{'-'*50}\nCONCLUSION:")
    if neg>0.6:
        L.append("  Malestar emocional detectado.")
        if pd.get('angry',0)>0.25: L.append("  -> Frustracion o enojo.")
        if pd.get('sad',0)>0.25: L.append("  -> Tristeza percibida.")
        if pd.get('fearful',0)>0.25: L.append("  -> Ansiedad detectada.")
        L.append("\n  ATENCION: Cuidar bienestar emocional.")
    elif pos>0.6:
        L.append("  Estado emocional positivo.")
        L.append("\n  Estado saludable.")
    elif neu>0.4:
        L.append("  Estado neutro/estable.")
    else:
        L.append("  Mezcla de emociones.")
        L.append(f"  -> {', '.join(f'{EMOJIS.get(e,chr(8226))} {e} ({p*100:.0f}%)' for e,p in se[:3])}")
    stress=(pd.get('angry',0)*0.9+pd.get('fearful',0)*0.95+pd.get('sad',0)*0.6+pd.get('disgust',0)*0.7+pd.get('surprised',0)*0.15)
    sl="BAJO" if stress<0.3 else "MEDIO" if stress<0.6 else "ALTO"
    L.append(f"\n{'-'*50}\nEstres: {sl} ({stress*100:.0f}%)")
    wb=max(0,min(100,pos*100+neu*50-neg*80+50))
    wl="Excelente" if wb>75 else "Bueno" if wb>50 else "Regular" if wb>30 else "Bajo"
    L.append(f"Bienestar: {wl} ({wb:.0f}/100)\n{'='*50}")
    return "\n".join(L)

# --- PREDICT ---
def predict(audio):
    if audio is None: return "No audio.", None, ""
    try:
        sr_in, data = audio
        if data.dtype != np.float32: data = data.astype(np.float32) / (np.max(np.abs(data)) + 1e-8)
        if len(data.shape) > 1: data = data.mean(axis=1)
        if CHAMP_INPUT == 'emb' and INF_PROC and INF_BACKBONE:
            d16 = librosa.resample(data, orig_sr=sr_in, target_sr=16000) if sr_in != 16000 else data
            d16 = np.pad(d16, (0, max(0, 48000 - len(d16))))[:48000]
            inp = INF_PROC(d16, sampling_rate=16000, return_tensors="pt", padding=True)
            inp = {k: v.to(DEVICE) for k, v in inp.items()}
            with torch.no_grad(): emb = INF_BACKBONE(**inp).last_hidden_state.mean(1).squeeze().cpu().numpy()
            tensor = torch.FloatTensor((emb - SCALER_EMB_MEAN) / (SCALER_EMB_SCALE + 1e-8)).unsqueeze(0).to(DEVICE)
        else:
            if sr_in != 22050: data = librosa.resample(data, orig_sr=sr_in, target_sr=22050)
            sf.write(os.path.join(BASE_DIR, '_t.wav'), data, 22050)
            ft = extract_features(os.path.join(BASE_DIR, '_t.wav'))
            if ft is None: return "Error features.", None, ""
            vec = np.array([ft.get(c, 0) for c in FEATURE_COLS], dtype=np.float32)
            tensor = torch.FloatTensor((vec - SCALER_MEAN) / (SCALER_SCALE + 1e-8)).unsqueeze(0).to(DEVICE)
        with torch.no_grad(): probs = F.softmax(model(tensor), 1).cpu().numpy()[0]
        pd_dict = {EMOTIONS[i]: float(probs[i]) for i in range(NUM_CLASSES)}
        di = np.argmax(probs)
        return (f"{EMOJIS[EMOTIONS[di]]} {EMOTIONS[di].upper()} ({probs[di]*100:.1f}%)",
                {f"{EMOJIS[e]} {e}": float(probs[i]) for i, e in enumerate(EMOTIONS)},
                diagnosis(pd_dict))
    except Exception as e: return f"Error: {e}", None, ""

# --- UI ---
ri = CONFIG.get('all_results', {})
acc_val = ri.get(CHAMPION_NAME, {}).get('test_acc', 0)
f1_val = ri.get(CHAMPION_NAME, {}).get('f1_weighted', 0)

with gr.Blocks(title="Diagnostico Emocional", theme=gr.themes.Soft(primary_hue="teal"),
               css=".gradio-container{max-width:1000px!important}") as demo:
    gr.Markdown(f"# Diagnostico Emocional por Nota de Voz\n### Universidad Panamericana - Deep Learning\n\n"
                f"**Modelo:** {CHAMPION_NAME} | **Accuracy:** {acc_val*100:.1f}% | **F1:** {f1_val:.3f}\n\n---\n"
                f"Graba o sube una nota de voz para analizar tus emociones.")
    with gr.Row():
        with gr.Column():
            audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Nota de Voz")
            btn = gr.Button("Analizar Emociones", variant="primary", size="lg")
            tbl = "### Ranking\n| # | Modelo | Acc |\n|---|---|---|\n"
            for i, (nm, r) in enumerate(sorted(ri.items(), key=lambda x: x[1]['test_acc'], reverse=True)):
                tbl += f"| {'**1**' if i==0 else i+1} | {nm} | {r['test_acc']:.3f} |\n"
            gr.Markdown(tbl)
        with gr.Column():
            res = gr.Textbox(label="Emocion", lines=1)
            prb = gr.Label(label="Emociones", num_top_classes=8)
            diag = gr.Textbox(label="Diagnostico", lines=25)
    btn.click(predict, inputs=audio_in, outputs=[res, prb, diag])
    gr.Markdown("\n---\n*Universidad Panamericana - Deep Learning - 2026*")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
