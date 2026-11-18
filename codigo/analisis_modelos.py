# -*- coding: utf-8 -*-
"""Modelado del proyecto: Respuesta sanitaria de Bolivia frente a COVID-19 (Ciencia de Datos).
Modelos: (1) k-means de perfiles de países, (2) series de tiempo + índice de presión sanitaria,
(3) clasificación de riesgo municipal. Uso: python analisis_modelos.py  (CSV en ./datos, figuras en ./fig)"""
import json, warnings, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
warnings.filterwarnings('ignore')
SEED=42; np.random.seed(SEED)
DATA=os.environ.get('DATA','datos'); FIG=os.environ.get('FIG','fig'); os.makedirs(FIG,exist_ok=True)
plt.rcParams.update({'font.family':'Liberation Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.25,'grid.linewidth':.6,'legend.frameon':False,'legend.fontsize':8,'savefig.dpi':220,'savefig.bbox':'tight'})
BOL='#B2182B'; GREY='#8C8C8C'; BLUE='#2166AC'; NAVY='#0B2A5B'; GREEN='#1B7837'
R={}
o=pd.read_csv(f'{DATA}/owid-covid-data.csv',parse_dates=['date'])
m=pd.read_csv(f'{DATA}/mun_covid_se30.csv'); m.columns=['coddepto','depto','codigo','municipio','se','conf','act','fall','rec']
C8=['Bolivia','Argentina','Brazil','Chile','Colombia','Ecuador','Paraguay','Peru']
ES={'Bolivia':'Bolivia','Argentina':'Argentina','Brazil':'Brasil','Chile':'Chile','Colombia':'Colombia','Ecuador':'Ecuador','Paraguay':'Paraguay','Peru':'Perú'}

# =====================================================  MODELO 1: PERFILES DE PAÍSES
last=o[o.date=='2020-09-19'].copy()
last=last[last.iso_code.notna()&~last.iso_code.astype(str).str.startswith('OWID')].set_index('location')
last['cfr']=last.total_deaths/last.total_cases*100
last['str_mean']=o.groupby('location').stringency_index.mean()
last['tests']=o.dropna(subset=['total_tests_per_thousand']).sort_values('date').groupby('location').tail(1).set_index('location').total_tests_per_thousand
F=['total_deaths_per_million','total_cases_per_million','cfr','tests','str_mean','hospital_beds_per_thousand','human_development_index','median_age']
FL=['Fallecidos/millón','Casos/millón','Letalidad (%)','Pruebas/1 000','Rigurosidad media','Camas/1 000','IDH','Edad mediana']
sa=last.loc[C8,F].copy(); sa.index=[ES[c] for c in sa.index]
Z=StandardScaler().fit_transform(sa)
res_sa={}
for k in (2,3,4):
    km=KMeans(k,n_init=50,random_state=SEED).fit(Z); res_sa[k]={'sil':float(silhouette_score(Z,km.labels_)),'labels':km.labels_.tolist()}
kbest=max(res_sa,key=lambda k:res_sa[k]['sil'])
ward=AgglomerativeClustering(kbest,linkage='ward').fit(Z)
R['sa']={'k_sil':{k:round(v['sil'],3) for k,v in res_sa.items()},'kbest':kbest,'labels':dict(zip(sa.index,res_sa[kbest]['labels'])),'ari_ward':float(adjusted_rand_score(res_sa[kbest]['labels'],ward.labels_)),'features':sa.round(2).reset_index().rename(columns={'index':'pais'}).to_dict('records')}
sa['cluster']=res_sa[kbest]['labels']
R['sa']['profile']=sa.groupby('cluster')[F].mean().round(2).reset_index().to_dict('records')
# muestra mundial de referencia
G=last[F].dropna().copy(); G_n=len(G)
ZG=StandardScaler().fit_transform(G)
resG={}
for k in range(2,9):
    km=KMeans(k,n_init=50,random_state=SEED).fit(ZG); resG[k]={'sil':float(silhouette_score(ZG,km.labels_)),'labels':km.labels_}
kG=max(resG,key=lambda k:resG[k]['sil'] if k>=3 else -1)  # k>=3 para evitar la partición trivial 2
ward=AgglomerativeClustering(kG,linkage='ward').fit(ZG)
G['cluster']=resG[kG]['labels']
R['global']={'n':G_n,'k_sil':{k:round(v['sil'],3) for k,v in resG.items()},'kbest':kG,'ari_ward':float(adjusted_rand_score(resG[kG]['labels'],ward.labels_)),
 'sizes':G.cluster.value_counts().sort_index().to_dict(),'profile':G.groupby('cluster')[F].mean().round(2).reset_index().to_dict('records')}
bcl=int(G.loc['Bolivia','cluster']); R['global']['bolivia_cluster']=bcl
R['global']['bolivia_cluster_members']=sorted(G[G.cluster==bcl].index.tolist())
R['global']['sa_clusters']={ES[c]:int(G.loc[c,'cluster']) for c in C8 if c in G.index}
dist=pd.Series(np.linalg.norm(ZG-ZG[list(G.index).index('Bolivia')],axis=1),index=G.index).drop('Bolivia').sort_values()
R['global']['bolivia_neighbors']=dist.head(6).round(2).to_dict()
# estabilidad: remuestreo de 50 corridas con semillas distintas
aris=[]
for s in range(50):
    km=KMeans(kG,n_init=1,random_state=s).fit(ZG); aris.append(adjusted_rand_score(resG[kG]['labels'],km.labels_))
R['global']['stab_ari_mean']=float(np.mean(aris)); R['global']['stab_ari_min']=float(np.min(aris))
# figuras: silueta y PCA
fig,axs=plt.subplots(1,2,figsize=(6.8,2.9))
axs[0].plot(list(res_sa),[res_sa[k]['sil'] for k in res_sa],marker='o',color=BLUE); axs[0].set_title('(a) Ocho países sudamericanos',loc='left',fontsize=9); axs[0].set_xticks(list(res_sa))
axs[1].plot(list(resG),[resG[k]['sil'] for k in resG],marker='o',color=GREEN); axs[1].set_title(f'(b) Muestra mundial (n = {G_n})',loc='left',fontsize=9); axs[1].set_xticks(list(resG))
for a in axs: a.set_xlabel('Número de grupos (k)'); a.set_ylabel('Coeficiente de silueta')
axs[1].axvline(kG,ls='--',color=BOL,lw=1); axs[0].axvline(kbest,ls='--',color=BOL,lw=1)
fig.tight_layout(); fig.savefig(f'{FIG}/m1_silueta.png'); plt.close()
pca=PCA(2,random_state=SEED).fit(ZG); P=pca.transform(ZG); R['global']['pca_var']=[round(float(v),3) for v in pca.explained_variance_ratio_]
load=pd.DataFrame(pca.components_.T,index=FL,columns=['PC1','PC2']).round(2); R['global']['pca_loadings']=load.reset_index().rename(columns={'index':'var'}).to_dict('records')
fig,ax=plt.subplots(figsize=(6.4,4.0)); cols=plt.cm.Set2(np.linspace(0,1,max(kG,3)))
for c in range(kG):
    idx=(G.cluster==c).values; ax.scatter(P[idx,0],P[idx,1],s=22,color=cols[c],alpha=.75,label=f'Grupo {c+1} (n = {idx.sum()})')
for c in C8:
    if c in G.index:
        i=list(G.index).index(c); ax.scatter(P[i,0],P[i,1],s=60 if c=='Bolivia' else 38,facecolor='none',edgecolor=BOL if c=='Bolivia' else '#222222',lw=1.6 if c=='Bolivia' else 1)
        ax.annotate(ES[c],(P[i,0],P[i,1]),xytext=(4,4),textcoords='offset points',fontsize=7.5,color=BOL if c=='Bolivia' else '#222222',fontweight='bold' if c=='Bolivia' else 'normal')
ax.set_xlabel(f'Componente 1 ({pca.explained_variance_ratio_[0]*100:.0f} % de la varianza)'); ax.set_ylabel(f'Componente 2 ({pca.explained_variance_ratio_[1]*100:.0f} %)'); ax.legend(loc='best')
fig.savefig(f'{FIG}/m1_pca.png'); plt.close()

# =====================================================  MODELO 2: SERIES DE TIEMPO
b=o[o.location=='Bolivia'].set_index('date').sort_index()
tc=b.total_cases.ffill()
wk_end=pd.date_range('2020-03-14','2020-09-19',freq='7D')
cum=tc.reindex(wk_end); y=cum.diff(); y.iloc[0]=cum.iloc[0]; y=y.astype(float)
n=len(y); R['ts']={'n_weeks':n,'first':str(wk_end[0].date()),'last':str(wk_end[-1].date()),'peak_week':str(y.idxmax().date()),'peak_value':float(y.max())}
H=4
def f_naive(tr,h): return np.repeat(tr[-1],h)
def f_ma3(tr,h): return np.repeat(np.mean(tr[-3:]),h)
def f_lin6(tr,h):
    x=np.arange(6); c=np.polyfit(x,tr[-6:],1); return np.maximum(np.polyval(c,np.arange(6,6+h)),0)
def f_poly2(tr,h):
    x=np.arange(len(tr)); c=np.polyfit(x,tr,2); return np.maximum(np.polyval(c,np.arange(len(tr),len(tr)+h)),0)
def f_holt(tr,h):
    mdl=ExponentialSmoothing(np.asarray(tr,float),trend='add',damped_trend=True,initialization_method='estimated').fit(); return np.maximum(mdl.forecast(h),0)
ORD=[(p,1,q) for p in range(3) for q in range(3)]
def f_arima(tr,h,return_order=False):
    ly=np.log1p(np.asarray(tr,float)); best=None
    for od in ORD:
        try:
            r=ARIMA(ly,order=od).fit()
            if best is None or r.aic<best[0]: best=(r.aic,od,r)
        except Exception: pass
    fc=np.expm1(best[2].forecast(h)); fc=np.maximum(fc,0)
    return (fc,best[1]) if return_order else fc
MODELS={'Ingenuo (último valor)':f_naive,'Media móvil de 3 semanas':f_ma3,'Regresión lineal (6 semanas)':f_lin6,'Regresión polinómica (grado 2)':f_poly2,'Suavizamiento de Holt amortiguado':f_holt,'ARIMA (p,1,q) sobre log':f_arima}
yy=y.values; origins=list(range(12,n-H+1))
rows=[]; pool={k:{'a':[],'f':[]} for k in MODELS}; orders=[]
for t in origins:
    tr=yy[:t]; te=yy[t:t+H]
    for k,fn in MODELS.items():
        if fn is f_arima:
            fc,od=fn(tr,H,True); orders.append(od)
        else: fc=fn(tr,H)
        pool[k]['a'].append(te); pool[k]['f'].append(fc)
tab=[]
for k in MODELS:
    A=np.concatenate(pool[k]['a']); Fc=np.concatenate(pool[k]['f'])
    mae=mean_absolute_error(A,Fc); rmse=float(np.sqrt(mean_squared_error(A,Fc))); r2=1-((A-Fc)**2).sum()/((A-A.mean())**2).sum()
    mape=float(np.mean(np.abs(A-Fc)/np.maximum(A,1))*100)
    # bloque final (último origen)
    Af=pool[k]['a'][-1]; Ff=pool[k]['f'][-1]
    tab.append({'modelo':k,'mae':round(mae,0),'rmse':round(rmse,0),'r2':round(float(r2),3),'mape':round(mape,1),'mae_final':round(mean_absolute_error(Af,Ff),0),'rmse_final':round(float(np.sqrt(mean_squared_error(Af,Ff))),0)})
R['ts']['metrics']=tab; R['ts']['n_origins']=len(origins); R['ts']['arima_orders']={str(o_):orders.count(o_) for o_ in sorted(set(orders))}
best=min(tab,key=lambda r:r['rmse'])['modelo']; R['ts']['best']=best
# bloque final de prueba (semanas 25-28)
t=origins[-1]; R['ts']['test_weeks']=[str(d.date()) for d in wk_end[t:t+H]]; R['ts']['test_actual']=[float(v) for v in yy[t:t+H]]
R['ts']['test_fc']={k:[round(float(v)) for v in pool[k]['f'][-1]] for k in MODELS}
# proyección final con el mejor modelo (+ ARIMA con intervalos)
fut=pd.date_range(wk_end[-1]+pd.Timedelta(days=7),periods=H,freq='7D')
fcs={k:fn(yy,H) for k,fn in MODELS.items() if k!='ARIMA (p,1,q) sobre log'}
fa,oda=f_arima(yy,H,True); fcs['ARIMA (p,1,q) sobre log']=fa; R['ts']['final_arima_order']=oda
ly=np.log1p(yy); ra=ARIMA(ly,order=oda).fit(); sf=ra.get_forecast(H)
ci80=np.expm1(sf.conf_int(alpha=.2)); ci95=np.expm1(sf.conf_int(alpha=.05))
# intervalos del mejor modelo: ARIMA si es el mejor; si no, bandas empíricas por error de validación
bestf=fcs[best]
errs=np.concatenate([(np.asarray(pool[best]['a'][i])-np.asarray(pool[best]['f'][i])) for i in range(len(origins))]); q10,q90=np.percentile(errs,[10,90]); q025,q975=np.percentile(errs,[2.5,97.5])
lo80=np.maximum(bestf+q10,0); hi80=bestf+q90; lo95=np.maximum(bestf+q025,0); hi95=bestf+q975
R['ts']['projection']={'weeks':[str(d.date()) for d in fut],'point':[round(float(v)) for v in bestf],'lo80':[round(float(v)) for v in lo80],'hi80':[round(float(v)) for v in hi80],'lo95':[round(float(v)) for v in lo95],'hi95':[round(float(v)) for v in hi95],'arima':[round(float(v)) for v in fa]}
R['ts']['last4']=[float(v) for v in yy[-4:]]
# figura: serie y backtest final
fig,ax=plt.subplots(figsize=(6.6,3.5))
ax.plot(wk_end,yy,color='#222222',lw=1.8,marker='o',ms=3,label='Observado')
ax.axvspan(wk_end[t],wk_end[-1]+pd.Timedelta(days=3),color='#EEEEEE',alpha=.7,lw=0); ax.text(wk_end[t]+pd.Timedelta(days=2),yy.max()*.02,'Bloque de prueba',fontsize=7.5,color='#555555',va='bottom')
pal={'Ingenuo (último valor)':'#999999','Regresión lineal (6 semanas)':'#E5B31B','Regresión polinómica (grado 2)':'#B279A2','Suavizamiento de Holt amortiguado':GREEN,'ARIMA (p,1,q) sobre log':BLUE}
for k,c in pal.items(): ax.plot(wk_end[t:t+H],pool[k]['f'][-1],color=c,lw=1.3,ls='--',marker='s',ms=2.5,label=k)
ax.set_ylabel('Casos nuevos por semana'); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,p:f'{v:,.0f}'.replace(',','\u00a0')))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b')); ax.legend(loc='upper left',fontsize=7)
fig.savefig(f'{FIG}/m2_backtest.png'); plt.close()
# figura: comparación de modelos
fig,ax=plt.subplots(figsize=(6.2,3.0)); names=[r['modelo'] for r in tab]; vals=[r['rmse'] for r in tab]
cols=[BOL if nme==best else GREY for nme in names]; ax.barh(names[::-1],vals[::-1],color=cols[::-1])
for i,v in enumerate(vals[::-1]): ax.text(v+80,i,f'{v:,.0f}'.replace(',','\u00a0'),va='center',fontsize=8)
ax.set_xlabel('RMSE de validación cronológica (casos por semana)'); ax.grid(axis='y',visible=False); ax.set_xlim(0,max(vals)*1.18)
fig.savefig(f'{FIG}/m2_rmse.png'); plt.close()
# figura: proyección
fig,ax=plt.subplots(figsize=(6.6,3.4))
ax.plot(wk_end,yy,color='#222222',lw=1.8,marker='o',ms=3,label='Observado (Our World in Data)')
ax.plot(fut,bestf,color=BOL,lw=1.8,marker='o',ms=3.5,label=f'Proyección: {best}')
ax.fill_between(fut,lo95,hi95,color=BOL,alpha=.12,label='Intervalo empírico 95 %'); ax.fill_between(fut,lo80,hi80,color=BOL,alpha=.25,label='Intervalo empírico 80 %')
ax.set_ylabel('Casos nuevos por semana'); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,p:f'{v:,.0f}'.replace(',','\u00a0')))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b')); ax.legend(loc='upper left',fontsize=7.5)
fig.savefig(f'{FIG}/m2_proyeccion.png'); plt.close()

# ---------- Índice de presión sanitaria (IPS): casos de 14 días por cama hospitalaria
ips={}; rows=[]
for c in C8:
    d=o[o.location==c].set_index('date').sort_index()
    pop=d.population.dropna().iloc[-1]; beds=d.hospital_beds_per_thousand.dropna().iloc[-1]*pop/1000
    s=d.new_cases.fillna(0).rolling(14).sum()/beds; ips[c]=s
    rows.append({'pais':ES[c],'camas_total':round(float(beds)),'ips_pico':round(float(s.max()),3),'fecha_pico':str(s.idxmax().date()),'ips_final':round(float(s.iloc[-1]),3)})
R['ips']={'tabla':rows}
bol_beds=[r['camas_total'] for r in rows if r['pais']=='Bolivia'][0]
# escenarios de capacidad sobre el pico observado de Bolivia y sobre la proyección
peak14=float(ips['Bolivia'].max()*bol_beds)
R['ips']['bolivia_peak_14d_cases']=peak14; R['ips']['bolivia_beds']=bol_beds
R['ips']['scenarios']=[{'camas_extra_pct':p,'ips_pico':round(peak14/(bol_beds*(1+p/100)),3)} for p in (0,25,50,100)]
proj14=float(bestf[:2].sum()); R['ips']['proj_14d_first']=round(proj14); R['ips']['proj_ips_first']=round(proj14/bol_beds,3)
R['ips']['last14_cases']=float(ips['Bolivia'].iloc[-1]*bol_beds)
# equivalencia: camas necesarias para igualar el IPS pico de otros países
peers_peak={r['pais']:r['ips_pico'] for r in rows}
R['ips']['camas_para_igualar_mediana_pares']=round(peak14/np.median([v for k,v in peers_peak.items() if k!='Bolivia']))
fig,axs=plt.subplots(1,2,figsize=(6.8,3.2),gridspec_kw={'width_ratios':[1.35,1]})
cp={'Bolivia':BOL,'Argentina':'#4C78A8','Brazil':'#F58518','Chile':'#54A24B','Colombia':'#B279A2','Ecuador':'#9D755D','Paraguay':'#72B7B2','Peru':'#E5B31B'}
for c in C8:
    axs[0].plot(ips[c].index,ips[c].values,color=cp[c],lw=2.4 if c=='Bolivia' else 1.1,label=ES[c],zorder=5 if c=='Bolivia' else 2)
axs[0].set_xlim(pd.Timestamp('2020-03-10'),pd.Timestamp('2020-09-20')); axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%b')); axs[0].set_ylabel('Casos de 14 días por cama'); axs[0].legend(ncol=2,fontsize=7); axs[0].set_title('(a) Serie',loc='left',fontsize=9)
sc=R['ips']['scenarios']; axs[1].bar([f'+{s["camas_extra_pct"]} %' if s['camas_extra_pct'] else 'Actual' for s in sc],[s['ips_pico'] for s in sc],color=[BOL]+[BLUE]*3)
for i,s in enumerate(sc): axs[1].text(i,s['ips_pico']+.03,f'{s["ips_pico"]:.2f}',ha='center',fontsize=8)
axs[1].set_ylabel('IPS en el pico de Bolivia'); axs[1].set_xlabel('Camas hospitalarias adicionales'); axs[1].set_title('(b) Escenarios de capacidad',loc='left',fontsize=9); axs[1].grid(axis='x',visible=False)
fig.tight_layout(); fig.savefig(f'{FIG}/m2_ips.png'); plt.close()

# =====================================================  MODELO 3: CLASIFICACIÓN DE RIESGO MUNICIPAL
THR=3.7; REF=27; MINC=30
p_conf=m.pivot(index='codigo',columns='se',values='conf'); p_fall=m.pivot(index='codigo',columns='se',values='fall')
s30=m[m.se==30].set_index('codigo'); s30=s30[s30.conf>=MINC].copy(); s30['cfr']=s30.fall/s30.conf*100; s30['y']=(s30.cfr>=THR).astype(int)
D=s30[['depto','y']].copy(); D['conf_ref']=p_conf[REF].reindex(D.index); D['fall_ref']=p_fall[REF].reindex(D.index)
D=D.dropna(); D=D[D.conf_ref>0]; D['cfr_ref']=D.fall_ref/D.conf_ref*100; D['log_conf']=np.log10(D.conf_ref)
R['clf']={'n':len(D),'pos':int(D.y.sum()),'neg':int((1-D.y).sum()),'thr':THR,'ref_se':REF,'min_conf':MINC}
dep=pd.get_dummies(D.depto,prefix='dep').astype(float)
XA=pd.concat([D[['log_conf']],dep],axis=1)                       # tamaño + departamento
XB=pd.concat([D[['log_conf','cfr_ref','fall_ref']],dep],axis=1)  # + información previa de letalidad
yv=D.y.values
cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=20,random_state=SEED)
def evalm(make,X,score_fn=None):
    M={'acc':[],'pre':[],'rec':[],'f1':[],'auc':[]}; cm=np.zeros((2,2))
    for tr,te in cv.split(X,yv):
        mdl=make(); mdl.fit(X.iloc[tr],yv[tr]); pr=mdl.predict(X.iloc[te]); ps=mdl.predict_proba(X.iloc[te])[:,1]
        M['acc'].append(accuracy_score(yv[te],pr)); M['pre'].append(precision_score(yv[te],pr,zero_division=0)); M['rec'].append(recall_score(yv[te],pr)); M['f1'].append(f1_score(yv[te],pr)); M['auc'].append(roc_auc_score(yv[te],ps)); cm+=confusion_matrix(yv[te],pr,labels=[0,1])
    return {k:(round(float(np.mean(v)),3),round(float(np.std(v)),3)) for k,v in M.items()},(cm/20).round(1).tolist()
mk={'Árbol de decisión (prof. 3)':lambda:DecisionTreeClassifier(max_depth=3,min_samples_leaf=5,random_state=SEED),
    'Bosque aleatorio':lambda:RandomForestClassifier(n_estimators=300,max_depth=4,min_samples_leaf=3,random_state=SEED,n_jobs=-1),
    'Regresión logística':lambda:LogisticRegression(max_iter=2000,C=1.0)}
out={}
for setname,X in (('A',XA),('B',XB)):
    for nm,fn in mk.items():
        if nm=='Regresión logística':
            from sklearn.pipeline import make_pipeline
            fn=(lambda f=fn:make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)))
        out[f'{setname}|{nm}']=evalm(fn,X)
# líneas base
base_major=max(yv.mean(),1-yv.mean())
pers=(D.cfr_ref>=THR).astype(int).values
R['clf']['baseline']={'mayoria_acc':round(float(base_major),3),'persistencia_acc':round(float(accuracy_score(yv,pers)),3),'persistencia_prec':round(float(precision_score(yv,pers)),3),'persistencia_rec':round(float(recall_score(yv,pers)),3),'persistencia_f1':round(float(f1_score(yv,pers)),3),'persistencia_auc':round(float(roc_auc_score(yv,D.cfr_ref.values)),3)}
R['clf']['results']={k:{'m':v[0],'cm':v[1]} for k,v in out.items()}
# importancia por permutación (bosque, conjunto B) con validación
rf=RandomForestClassifier(n_estimators=300,max_depth=4,min_samples_leaf=3,random_state=SEED,n_jobs=-1)
imps=[]
for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=SEED).split(XB,yv):
    rf.fit(XB.iloc[tr],yv[tr]); pi=permutation_importance(rf,XB.iloc[te],yv[te],n_repeats=10,random_state=SEED,scoring='roc_auc'); imps.append(pi.importances_mean)
imp=pd.Series(np.mean(imps,axis=0),index=XB.columns).sort_values(ascending=False); R['clf']['importance']=imp.round(3).head(8).to_dict()
# figuras
fig,axs=plt.subplots(1,2,figsize=(6.8,3.0))
t=imp.head(7)[::-1]; lab={'log_conf':'log10 confirmados (SE 27)','cfr_ref':'Letalidad SE 27','fall_ref':'Fallecidos SE 27'}
axs[0].barh([lab.get(i,i.replace('dep_','Dpto. ').title()) for i in t.index],t.values,color=BLUE); axs[0].set_xlabel('Caída del AUC al permutar'); axs[0].set_title('(a) Importancia de variables',loc='left',fontsize=9); axs[0].grid(axis='y',visible=False)
names=['A|Bosque aleatorio','B|Bosque aleatorio','B|Regresión logística']; vals=[out[k][0]['f1'][0] for k in names]; sds=[out[k][0]['f1'][1] for k in names]
labs=['A: bosque\n(tamaño y dpto.)','B: bosque\n(+ letalidad\nprevia)','B: regresión\nlogística']
axs[1].bar(labs,vals,yerr=sds,color=[GREY,BOL,BLUE],capsize=3); axs[1].axhline(R['clf']['baseline']['persistencia_f1'],ls='--',color='#222222',lw=1); axs[1].text(-0.45,R['clf']['baseline']['persistencia_f1']+.025,'Línea base (persistencia): F1 = 0.79',ha='left',fontsize=7.5)
axs[1].set_ylim(0,1.05); axs[1].set_ylabel('F1 (validación cruzada repetida)'); axs[1].set_title('(b) Desempeño',loc='left',fontsize=9); axs[1].tick_params(axis='x',labelsize=7); axs[1].grid(axis='x',visible=False)
fig.tight_layout(); fig.savefig(f'{FIG}/m3_clasificacion.png'); plt.close()
# matriz de confusión media del bosque B
cm=np.array(out['B|Bosque aleatorio'][1]); fig,ax=plt.subplots(figsize=(3.3,2.9))
ax.imshow(cm,cmap='Blues'); ax.grid(False)
for i in range(2):
    for j in range(2): ax.text(j,i,f'{cm[i,j]:.1f}',ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else '#222222',fontsize=11)
ax.set_xticks([0,1]); ax.set_xticklabels(['Bajo','Alto']); ax.set_yticks([0,1]); ax.set_yticklabels(['Bajo','Alto']); ax.set_xlabel('Predicho'); ax.set_ylabel('Real'); ax.set_title('Promedio por partición',fontsize=8.5)
fig.savefig(f'{FIG}/m3_confusion.png'); plt.close()
json.dump(R,open('resultados.json','w'),indent=1,default=lambda x:float(x) if isinstance(x,(np.floating,np.integer)) else str(x))
print('OK')

# =====================================================  FIGURAS ADICIONALES (diagrama metodológico y tablero)
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
def flujo():
    fig,ax=plt.subplots(figsize=(7.0,5.0)); ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis('off')
    def box(x,y,w,h,t,fc,ec=NAVY,fs=8,bold=False,ha='center'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.3,rounding_size=1.2',fc=fc,ec=ec,lw=1.0))
        ax.text(x+(w/2 if ha=='center' else 1.5),y+h/2,t,ha=ha,va='center',fontsize=fs,fontweight='bold' if bold else 'normal',color='#1A1A1A',linespacing=1.3)
    rows=[('1. Comprensión del negocio','#DCE6F4','Problema, objetivos y criterios de éxito; riesgos y recursos; plan del proyecto.'),
          ('2. Comprensión de los datos','#DCE6F4','Dos fuentes abiertas (Ministerio de Salud y Deportes; Our World in Data); exploración y control de calidad.'),
          ('3. Preparación de los datos','#DCE6F4','Limpieza, identificador único, semanas epidemiológicas → fechas, indicadores derivados, faltantes y atípicos.'),
          ('4. Modelado','#F8E3D3','k-means (perfiles de países); Holt, ARIMA y regresión (series de tiempo); árbol, bosque y logística (riesgo municipal).'),
          ('5. Evaluación','#F8E3D3','Silueta y ARI; MAE, RMSE y R² con validación cronológica; exactitud, F1 y AUC con validación cruzada repetida.'),
          ('6. Interpretación y comunicación','#E1EFD9','Respuesta a la pregunta, conclusiones, recomendaciones, informe, defensa, código y tablero.')]
    h=11.5; gap=3.2; y=92-h
    for i,(t,c,d_) in enumerate(rows):
        import textwrap; box(1,y,29,h,textwrap.fill(t,22),c,bold=True,fs=7.8); box(33,y,66,h,textwrap.fill(d_,64),'white',ec='#999999',fs=7.3,ha='left')
        ax.add_patch(FancyArrowPatch((30.6,y+h/2),(32.4,y+h/2),arrowstyle='-|>',mutation_scale=8,color='#444444',lw=1))
        if i<len(rows)-1: ax.add_patch(FancyArrowPatch((15.5,y-0.2),(15.5,y-gap+0.4),arrowstyle='-|>',mutation_scale=8,color='#444444',lw=1))
        y-=h+gap
    ax.add_patch(FancyArrowPatch((99.8,92-h-(h+gap)*5+h/2),(99.8,92-h/2),connectionstyle='arc3,rad=0',arrowstyle='-',color='#B2182B',lw=0))
    ax.text(50,2.2,'Proceso iterativo: los hallazgos de cada fase pueden devolver el trabajo a fases anteriores.',ha='center',fontsize=7.6,style='italic',color='#444444')
    fig.savefig(f'{FIG}/flujo_metodologico.png'); plt.close()
flujo()
def tablero():
    fig=plt.figure(figsize=(7.2,6.2)); gs=fig.add_gridspec(2,2,hspace=.42,wspace=.32)
    a=fig.add_subplot(gs[0,0]); s=pd.DataFrame(R['ips']['tabla']).sort_values('ips_pico'); a.barh(s.pais,s.ips_pico,color=[BOL if p=='Bolivia' else GREY for p in s.pais])
    for i,v in enumerate(s.ips_pico): a.text(v+.04,i,f'{v:.2f}',va='center',fontsize=7.5)
    a.set_title('A. Presión máxima sobre camas (IPS)',loc='left',fontsize=9,color=NAVY,fontweight='bold'); a.set_xlabel('Casos de 14 días por cama',fontsize=8); a.grid(axis='y',visible=False); a.set_xlim(0,3.2)
    b_=fig.add_subplot(gs[0,1]); 
    for c in range(kG):
        idx=(G.cluster==c).values; b_.scatter(P[idx,0],P[idx,1],s=12,color=plt.cm.Set2(np.linspace(0,1,max(kG,3)))[c],alpha=.7)
    i=list(G.index).index('Bolivia'); b_.scatter(P[i,0],P[i,1],s=70,facecolor='none',edgecolor=BOL,lw=1.8); b_.annotate('Bolivia',(P[i,0],P[i,1]),xytext=(5,5),textcoords='offset points',fontsize=8,color=BOL,fontweight='bold')
    b_.set_title(f'B. Perfiles de respuesta ({kG} grupos)',loc='left',fontsize=9,color=NAVY,fontweight='bold'); b_.set_xlabel('Componente 1',fontsize=8); b_.set_ylabel('Componente 2',fontsize=8)
    c_=fig.add_subplot(gs[1,0]); c_.plot(wk_end,yy,color='#222222',lw=1.5); c_.plot(fut,bestf,color=BOL,lw=1.8,marker='o',ms=3); c_.fill_between(fut,lo80,hi80,color=BOL,alpha=.25)
    c_.set_title('C. Casos semanales y proyección',loc='left',fontsize=9,color=NAVY,fontweight='bold'); c_.xaxis.set_major_formatter(mdates.DateFormatter('%b')); c_.set_ylabel('Casos por semana',fontsize=8); c_.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,p:f'{v/1000:.0f} mil'))
    d_=fig.add_subplot(gs[1,1]); t_=imp.head(5)[::-1]; d_.barh([lab.get(i_,i_.replace('dep_','Dpto. ').title()) for i_ in t_.index],t_.values,color=BLUE)
    d_.set_title('D. Qué predice la letalidad alta',loc='left',fontsize=9,color=NAVY,fontweight='bold'); d_.set_xlabel('Caída del AUC al permutar',fontsize=8); d_.grid(axis='y',visible=False); d_.tick_params(axis='y',labelsize=7.5)
    fig.savefig(f'{FIG}/tablero_resumen.png'); plt.close()
tablero()
print('FIGURAS OK')
