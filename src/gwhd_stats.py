import pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np
meta=pd.read_csv('77264cdf-metadata_dataset.csv',sep=';'); meta['key']=meta.name.str.lower()
meta['development_stage']=meta.development_stage.str.replace('Post-Flowering','Post-flowering').replace({'multiple':'Multiple'})
parts=[]
for s,f in [('Train','19410d63-competition_train.csv'),('Validation','2cb383b4-competition_val.csv'),('Test','81e52d93-competition_test.csv')]:
    d=pd.read_csv(f); d['split']=s
    d['n']=d.BoxesString.apply(lambda b:0 if pd.isna(b) or str(b).strip()=='no_box' else len([x for x in str(b).split(';') if x.strip()]))
    parts.append(d)
df=pd.concat(parts); df['key']=df.domain.str.lower()
m=df.merge(meta,on='key',how='left'); assert m.country.isna().sum()==0
g=m.groupby('split',sort=False)
tab=pd.DataFrame({'images':g.size(),'heads':g.n.sum(),'mean':g.n.mean().round(1),'sd':g.n.std().round(1),'min':g.n.min(),'max':g.n.max(),
 'domains':g.domain.nunique(),'countries':g.country.nunique(),'stages':g.development_stage.nunique()})
tab.loc['Total']=[len(m),m.n.sum(),round(m.n.mean(),1),round(m.n.std(),1),m.n.min(),m.n.max(),m.domain.nunique(),m.country.nunique(),m.development_stage.nunique()]
print(tab.to_string())
for s in ['Train','Validation','Test']:
    x=m[m.split==s]; print(s,', '.join(sorted(x.country.unique())),'|',', '.join(sorted(x.development_stage.unique())))
print(m.groupby('development_stage').size())
# figure
C={'Train':'#2a78d6','Validation':'#eb6834','Test':'#1baf7a'}
plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#52514e','axes.labelcolor':'#0b0b0b'})
fig,ax=plt.subplots(1,3,figsize=(11,3.4),gridspec_kw={'width_ratios':[1.5,1,1.3]})
ct=m.groupby(['country','split']).size().unstack(fill_value=0)[['Train','Validation','Test']]
ct=ct.loc[ct.sum(1).sort_values().index]
left=np.zeros(len(ct))
for s in ct.columns:
    ax[0].barh(ct.index,ct[s],left=left,color=C[s],label=s,height=0.7,edgecolor='white',linewidth=1); left+=ct[s].values
ax[0].set_xlabel('Number of images'); ax[0].set_title('(a) Images per country and split',loc='left',fontsize=10)
ax[0].legend(frameon=False,loc='lower right')
order=['Post-flowering','Filling','Filling - Ripening','Ripening','Multiple']
st=m.groupby(['development_stage','split']).size().unstack(fill_value=0).reindex(order)[['Train','Validation','Test']]
xx=np.arange(len(order)); w=0.27
for i,s in enumerate(st.columns): ax[1].bar(xx+(i-1)*w,st[s],w,color=C[s],label=s,edgecolor='white',linewidth=1)
ax[1].set_xticks(xx); ax[1].set_xticklabels(['Post-\nflower.','Filling','Filling–\nripen.','Ripen-\ning','Multi-\nple'],fontsize=8)
ax[1].set_ylabel('Number of images'); ax[1].set_title('(b) Growth stage',loc='left',fontsize=10)
bins=np.arange(0,200,5)
for s in ['Train','Validation','Test']:
    ax[2].hist(m[m.split==s].n,bins=bins,histtype='step',linewidth=2,color=C[s],label=s,density=True)
ax[2].set_xlabel('Annotated wheat heads per image'); ax[2].set_ylabel('Density'); ax[2].set_title('(c) Head-density distribution',loc='left',fontsize=10)
ax[2].legend(frameon=False)
for a in ax: a.grid(axis='x' if a is ax[0] else 'y',color='#e6e5e1',linewidth=0.6); a.set_axisbelow(True)
fig.tight_layout(); fig.savefig('rev/fig_gwhd_diversity.pdf'); fig.savefig('rev/fig_gwhd_diversity.png',dpi=300)
