# -*- coding: utf-8 -*-
"""Análisis exploratorio (sección 9). Genera estadísticas y figuras del EDA.
Uso: DATA=datos FIG=figuras python eda_analisis.py"""
import os, json
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.ticker as mt
import seaborn as sns
from scipy import stats
DATA=os.environ.get('DATA','datos'); FIG=os.environ.get('FIG','figuras'); os.makedirs(FIG,exist_ok=True)
plt.rcParams.update({'font.family':'Liberation Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.25,'legend.frameon':False,'savefig.dpi':220,'savefig.bbox':'tight'})
BOL='#B2182B'; GREY='#8C8C8C'; BLUE='#2166AC'; S={}
m=pd.read_csv(f'{DATA}/mun_covid_se30.csv'); m.columns=['coddepto','depto','codigo','municipio','se','conf','act','fall','rec']
m['depto']=m.depto.str.title().replace({'Potosi':'Potosí'})

# ---- calidad de datos
S['registros']=len(m); S['municipios']=int(m.codigo.nunique()); S['duplicados']=int(m.duplicated(['codigo','se']).sum()); S['nulos']=int(m.isna().sum().sum())
S['identidad_violada']=int(((m.conf-m.act-m.fall-m.rec)!=0).sum()); S['suma_filas_fallecidos']=int(m.fall.sum())
perse=m.groupby('se')[['conf','act','fall','rec']].sum().reset_index()
s30=m[m.se==30].copy(); s30['cfr']=s30.fall/s30.conf*100
S['total_se30']=s30[['conf','act','fall','rec']].sum().to_dict()

# ---- figura 1: evolución nacional
fig,ax=plt.subplots(figsize=(6.4,3.4))
for k,(lab,c) in {'conf':('Confirmados','#222222'),'act':('Activos',BLUE),'rec':('Recuperados','#1B7837'),'fall':('Fallecidos',BOL)}.items(): ax.plot(perse.se,perse[k],marker='o',ms=3.5,color=c,label=lab,lw=1.6)
ax.axvspan(25,27,color='#DDDDDD',alpha=.5,lw=0); ax.set_xlabel('Semana epidemiológica (2020)'); ax.set_ylabel('Casos acumulados'); ax.set_xticks(range(20,31)); ax.legend(); fig.savefig(f'{FIG}/eda_evolucion_nacional.png'); plt.close()

# ---- figura 2: histogramas
fig,axs=plt.subplots(1,2,figsize=(6.6,3.0))
axs[0].hist(np.log10(s30.conf),bins=np.arange(0,4.6,.25),color=BLUE,edgecolor='white'); axs[0].set_xlabel('Casos confirmados por municipio (escala log)')
axs[1].hist(s30.fall,bins=[0,1,2,3,4,5,10,20,50,100,1000],color=BOL,edgecolor='white'); axs[1].set_xscale('symlog',linthresh=5); axs[1].set_xlabel('Fallecidos por municipio')
fig.tight_layout(); fig.savefig(f'{FIG}/eda_histogramas.png'); plt.close()

# ---- departamentos
dep=s30.groupby('depto').agg(conf=('conf','sum'),fall=('fall','sum'),rec=('rec','sum')).reset_index(); dep['cfr']=dep.fall/dep.conf*100
S['departamentos']=dep.round(2).to_dict('records')
fig,axs=plt.subplots(1,2,figsize=(6.6,3.3))
d2=dep.sort_values('conf'); axs[0].barh(d2.depto,d2.conf,color=BLUE); axs[0].barh(d2.depto,d2.fall,color=BOL)
d3=dep.sort_values('cfr'); axs[1].barh(d3.depto,d3.cfr,color=[BOL if v>3.7 else GREY for v in d3.cfr]); axs[1].set_xlabel('Letalidad (%)')
fig.tight_layout(); fig.savefig(f'{FIG}/eda_departamentos.png'); plt.close()

# ---- letalidad municipal (>= 50 casos) y prueba de Kruskal-Wallis
c50=s30[s30.conf>=50]; grp=[g.cfr.values for _,g in c50.groupby('depto') if len(g)>=3]
S['kruskal']=dict(zip(['H','p'],map(float,stats.kruskal(*grp)))); S['mediana_cfr_50']=float(c50.cfr.median())
fig,ax=plt.subplots(figsize=(6.4,3.3)); sns.boxplot(data=c50,x='depto',y='cfr',color='#F4C6C9',fliersize=0,ax=ax); sns.stripplot(data=c50,x='depto',y='cfr',color=BOL,size=3,ax=ax)
plt.setp(ax.get_xticklabels(),rotation=30,ha='right'); fig.savefig(f'{FIG}/eda_letalidad_municipal.png'); plt.close()

# ---- correlaciones
cor=s30[['conf','act','fall','rec']].corr(method='spearman'); S['spearman']=cor.round(2).to_dict()
fig,ax=plt.subplots(figsize=(3.9,3.2)); sns.heatmap(cor,annot=True,fmt='.2f',cmap='Blues',vmin=0,vmax=1,ax=ax); fig.savefig(f'{FIG}/eda_correlaciones.png'); plt.close()

# ---- Our World in Data
o=pd.read_csv(f'{DATA}/owid-covid-data.csv',parse_dates=['date'])
ES={'Bolivia':'Bolivia','Argentina':'Argentina','Brazil':'Brasil','Chile':'Chile','Colombia':'Colombia','Ecuador':'Ecuador','Paraguay':'Paraguay','Peru':'Perú'}
x8=o[o.location.isin(ES)].copy(); x8['pais']=x8.location.map(ES)
fig,ax=plt.subplots(figsize=(6.5,3.5))
for c in ES.values():
    d=x8[x8.pais==c]; ax.plot(d.date,d.new_cases_smoothed_per_million,lw=2.6 if c=='Bolivia' else 1.2,color=BOL if c=='Bolivia' else None,label=c)
ax.set_ylabel('Nuevos casos por millón (media móvil 7 días)'); ax.legend(ncol=4); fig.savefig(f'{FIG}/eda_casos_por_millon.png'); plt.close()
snap=x8[x8.date=='2020-09-19'].set_index('pais'); snap['cfr']=snap.total_deaths/snap.total_cases*100
S['indicadores_acumulados']=snap[['total_cases','total_deaths','total_cases_per_million','total_deaths_per_million','cfr']].round(1).reset_index().to_dict('records')
fig,axs=plt.subplots(1,2,figsize=(6.6,3.2))
sd=snap.sort_values('total_deaths_per_million'); axs[0].barh(sd.index,sd.total_deaths_per_million,color=[BOL if i=='Bolivia' else GREY for i in sd.index]); axs[0].set_xlabel('Fallecidos por millón')
sc=snap.sort_values('cfr'); axs[1].barh(sc.index,sc.cfr,color=[BOL if i=='Bolivia' else GREY for i in sc.index]); axs[1].set_xlabel('Letalidad aparente (%)')
fig.tight_layout(); fig.savefig(f'{FIG}/eda_mortalidad_letalidad.png'); plt.close()
json.dump(S,open('resultados_eda.json','w'),indent=1,default=str); print('EDA OK')
