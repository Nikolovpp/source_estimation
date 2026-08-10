#!/usr/bin/env python3
"""The three narrative figures for the GC route comparison.

These answer, in order, the questions the analysis was designed around:

    story1  Do the parametric and state-space estimators differ at all?
    story2  If so, does it depend on window, order, inverse or task?
    story3  Config F (hypothesis-based conditioning) as the PRIMARY analysis:
            what do the pre-registered dual-stream triples show?

They are the readable path through the exhaustive material. The 24 ``cov_*.pdf``
sweeps and ``fig1``-``fig10`` remain the supplementary reference and are built by
``plot_granger_routes.py``; nothing here replaces them.

    conda activate mne && python plot_granger_story.py

Writes into <GC_routes>/_figures/.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from scipy.stats import ttest_rel

from granger_routes_stats import load_dir, bh_fdr
from run_granger_routes import PRIMARY_TRIPLES

R = ('/mnt/r/phd_thesis/Research/SpeechProduction/EEG/derivatives/'
     'source_estimation/GC_routes')
OUT = os.path.join(R, '_figures')

sns.set_theme(context='notebook', style='white', font_scale=0.9,
              rc={'figure.facecolor': 'white', 'axes.titlepad': 8})
_P = sns.color_palette('colorblind')
PAR, SS = _P[3], _P[0]          # parametric / state-space, used consistently
INK, MUT = '#222222', _P[7]
UNC, CON = _P[7], _P[0]   # unconditioned vs conditioned, story3
BANDS = ['theta', 'low_beta', 'high_beta']
BL = {'theta': 'theta', 'low_beta': 'low beta', 'high_beta': 'high beta'}
BC = dict(zip(BANDS, sns.color_palette('crest', 3)))


def G(tag, sub, task='overtProd', stim='prodDiff', meth='dSPM'):
    """Load one config directory."""
    return load_dir(os.path.join(R, task, meth, 'custom', tag, sub, stim))


def sh(roi):
    """Short ROI label for axes: 'awfa-lh' -> 'awfa'."""
    return roi.replace('-lh', '')


def cat(st, pre, bi):
    """Every subject x window value for one measure and band, across all edges."""
    return np.concatenate([st[k][:, bi].ravel()
                           for k in st if k.startswith(pre + '__')])


def story1():
    """Q1 - do the estimators differ? Model schematic + the two scatters."""
    import pandas as pd
    def box(ax,x,y,w,h,txt,fc,ec,fs=10,tc=INK):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012',
                                    fc=fc,ec=ec,lw=1.4,zorder=3))
        ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,color=tc,zorder=4)

    def arrow(ax,p0,p1,col,lw=1.6,ls='-'):
        ax.annotate('',xy=p1,xytext=p0,zorder=2,
                    arrowprops=dict(arrowstyle='-|>',color=col,lw=lw,ls=ls,
                                    shrinkA=2,shrinkB=3,mutation_scale=13))

    fig,ax=plt.subplots(1,3,figsize=(15.0,4.8),
                        gridspec_kw={'width_ratios':[1.25,1,1]})
    A=ax[0]; A.set_xlim(0,1); A.set_ylim(-0.42,1.0); A.axis('off')
    A.set_title('(a)  the two models GC compares',fontsize=11,loc='left',color=INK)

    # FULL model
    A.text(0.02,0.93,'FULL',fontsize=10.5,color=INK,weight='bold')
    box(A,0.03,0.70,0.20,0.13,"$y$ past",'#eef3f8',SS)
    box(A,0.03,0.55,0.20,0.13,"$x$ past",'#fdece6',PAR)
    box(A,0.52,0.625,0.17,0.13,"$y_t$",'white',INK)
    arrow(A,(0.235,0.765),(0.515,0.71),SS)
    arrow(A,(0.235,0.615),(0.515,0.665),PAR)
    A.text(0.72,0.69,r'$\sigma^2_{\rm full}$',fontsize=12,color=INK,va='center')

    # REDUCED model
    A.text(0.02,0.44,'REDUCED',fontsize=10.5,color=INK,weight='bold')
    box(A,0.03,0.21,0.20,0.13,"$y$ past",'#eef3f8',SS)
    box(A,0.03,0.06,0.20,0.13,"$x$ past",'#f4f4f4','#cccccc',tc='#bbbbbb')
    box(A,0.52,0.135,0.17,0.13,"$y_t$",'white',INK)
    arrow(A,(0.235,0.275),(0.515,0.22),SS)
    A.plot([0.13,0.13],[0.19,0.21],color='#bbbbbb',lw=1)
    A.text(0.29,0.125,'✕',fontsize=15,color='#b2182b',ha='center',va='center')
    A.text(0.72,0.20,r'$\sigma^2_{\rm red}$',fontsize=12,color=INK,va='center')

    A.plot([0.0,1.0],[0.50,0.50],color=MUT,lw=0.8,ls=(0,(3,3)))
    A.plot([0.0,1.0],[-0.02,-0.02],color=MUT,lw=0.8,ls=(0,(3,3)))
    A.text(0.5,-0.17,r'$F_{x\to y}=\ln\dfrac{\sigma^2_{\rm red}}{\sigma^2_{\rm full}}$',
           fontsize=15,ha='center',va='center',color=INK)
    A.text(0.5,-0.36,r'parametric: $\sigma^2_{\rm red}$ FITTED      '
           r'state-space: $\sigma^2_{\rm red}$ DERIVED',
           fontsize=9.5,ha='center',va='center',color=MUT)

    # the two scatters
    s,b,w,st=load_dir(f'{R}/overtProd/dSPM/custom/A_canonical/win60ms_order6_fs200_pc1/prodDiff')
    bi=b.index('theta')
    cat=lambda pre: np.concatenate([st[k][:,bi].ravel() for k in st if k.startswith(pre+'__')])
    for axx,meas,col,ttl in ((ax[1],'m1',SS,'(b)  conditional spectral GC'),
                             (ax[2],'m2',PAR,'(c)  pairwise time-domain GC')):
        p_,s_=cat(meas+'_par'),cat(meas+'_ss')
        ok=np.isfinite(p_)&np.isfinite(s_)
        hi=np.nanpercentile(np.r_[p_[ok],s_[ok]],99.5)
        # a couple of extreme parametric outliers would otherwise squash everything
        lo=min(0.0,np.nanpercentile(p_[ok],0.5))
        span=max(hi-0.0,0.0-lo); lo,hi=lo-0.03*span,hi+0.03*span
        axx.plot([lo,hi],[lo,hi],color=MUT,lw=1.2,ls=(0,(4,3)),zorder=1)
        neg=p_[ok]<0
        axx.scatter(p_[ok][~neg],s_[ok][~neg],s=2.5,color=col,alpha=0.16,lw=0,zorder=3)
        if neg.any():
            axx.scatter(p_[ok][neg],s_[ok][neg],s=4,color='#b2182b',alpha=0.35,lw=0,zorder=4)
        axx.axvline(0,color=INK,lw=1.3,zorder=5)
        axx.set(xlabel='parametric',ylabel='state-space',xlim=(lo,hi),ylim=(lo,hi),title=ttl)
        axx.set_aspect('equal',adjustable='box')
        r=np.nanmean(p_[ok])/np.nanmean(s_[ok])
        bb=dict(boxstyle='round,pad=0.3',fc='white',ec='none',alpha=0.88)
        has_neg = neg.mean() > 0.01
        axx.text(0.97, 0.13 if has_neg else 0.04, f'ratio {r:.2f}',
                 transform=axx.transAxes, va='bottom', ha='right',
                 fontsize=11, color=col, bbox=bb, zorder=8)
        if has_neg:
            axx.text(0.97, 0.04, f'{100*neg.mean():.0f}% negative',
                     transform=axx.transAxes, va='bottom', ha='right',
                     fontsize=10.5, color='#b2182b', bbox=bb, zorder=8)
        sns.despine(ax=axx); axx.grid(color=MUT,alpha=0.18,lw=0.6); axx.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f'{OUT}/story1_do_estimators_differ.png',dpi=180,bbox_inches='tight',facecolor='white')


def story2():
    """Q2 - window, order, inverse and task: which one drives the difference?"""
    import pandas as pd
    # ══════════ Q2 — does it depend on the parameters? ══════════
    fig,ax=plt.subplots(2,2,figsize=(12.6,8.0))
    wins=[40,60,80,120]
    A=ax[0][0]
    for meas,col,lab in (('m1',SS,'conditional spectral'),('m2',PAR,'time-domain')):
        y=[]
        for wm in wins:
            _,bb,_,stt=G('B_winsweep',f'win{wm}ms_order6_fs200_pc1')
            bi2=bb.index('theta')
            y.append(np.nanmean(cat(stt,meas+'_par',bi2))/np.nanmean(cat(stt,meas+'_ss',bi2)))
        A.plot(wins,y,color=col,lw=2.3,marker='o',ms=6,label=lab,zorder=3)
        for a_,b_ in zip(wins,y): A.annotate(f'{b_:.2f}',(a_,b_),textcoords='offset points',
                                              xytext=(0,-15),ha='center',fontsize=8.5,color=col)
    A.axhline(1,color=INK,lw=1.2,ls=(0,(4,3))); A.axhline(0,color='#b2182b',lw=1.2)
    A.set(xlabel='sliding window (ms), order 6',ylabel='parametric ÷ state-space',
          title='(a) WINDOW — the time-domain estimator fails\nbelow ~80 ms, and goes negative at 40 ms')
    A.legend(fontsize=8.5,frameon=False,loc='lower right')
    B=ax[0][1]
    for tag,subs_,xs,ls,mk in (('C_ordersweep_win60',[2,4,6,8],[2,4,6,8],'-','o'),
                               ('C_ordersweep_win120',[4,8,12,16,20],[4,8,12,16,20],'--','s')):
        wm=60 if '60' in tag else 120
        for meas,col in (('m1',SS),('m2',PAR)):
            y=[]
            for o in subs_:
                _,bb,_,stt=G(tag,f'win{wm}ms_order{o}_fs200_pc1'); bi2=bb.index('theta')
                y.append(np.nanmean(cat(stt,meas+'_par',bi2))/np.nanmean(cat(stt,meas+'_ss',bi2)))
            B.plot(xs,y,color=col,lw=2.1,ls=ls,marker=mk,ms=5,zorder=3)
    B.axhline(1,color=INK,lw=1.2,ls=(0,(4,3)))
    for h,l in ((dict(color=SS,lw=2),'conditional spectral'),(dict(color=PAR,lw=2),'time-domain'),
                (dict(color=INK,lw=1.6,ls='-'),'60 ms window'),(dict(color=INK,lw=1.6,ls='--'),'120 ms window')):
        B.plot([],[],**h,label=l)
    B.set(xlabel='model order',ylabel='parametric ÷ state-space',
          title='(b) ORDER — at 120 ms nothing breaks.\nThe window is the cause, not the order')
    B.legend(fontsize=8,frameon=False,loc='lower left',ncol=2)
    C=ax[1][0]; xs=np.arange(len(wins))
    for k,(meth,hatch) in enumerate((('dSPM',None),('LCMV','//'))):
        y=[]
        for wm in wins:
            _,bb,_,stt=G('B_winsweep',f'win{wm}ms_order6_fs200_pc1',meth=meth); bi2=bb.index('theta')
            y.append(np.nanmean(cat(stt,'m1_par',bi2))/np.nanmean(cat(stt,'m1_ss',bi2)))
        C.bar(xs+(k-0.5)*0.36,y,0.36,color=SS,alpha=(1.0 if k==0 else 0.55),
              hatch=hatch,edgecolor='white',label=meth,zorder=3)
    C.set(xticks=xs,xticklabels=[f'{w} ms' for w in wins],ylim=(0.85,1.0),
          ylabel='parametric ÷ state-space (m1)',
          title='(c) SOURCE ESTIMATOR — dSPM vs LCMV differ by ~1%.\nThe inverse is not what drives this')
    C.legend(fontsize=8.5,frameon=False)
    D=ax[1][1]
    for k,(task,stim,mk) in enumerate((('overtProd','prodDiff','o'),('perception','percDiff','^'))):
        for meas,col in (('m1',SS),('m2',PAR)):
            y=[]
            for o in [2,4,6,8]:
                _,bb,_,stt=G('C_ordersweep_win60',f'win60ms_order{o}_fs200_pc1',task=task,stim=stim)
                bi2=bb.index('theta')
                y.append(np.nanmean(cat(stt,meas+'_par',bi2))/np.nanmean(cat(stt,meas+'_ss',bi2)))
            D.plot([2,4,6,8],y,color=col,lw=2.0,marker=mk,ms=6,
                   ls=('-' if k==0 else '--'),zorder=3)
    D.axhline(1,color=INK,lw=1.2,ls=(0,(4,3)))
    D.plot([],[],color=INK,marker='o',ls='-',label='overtProd / prodDiff')
    D.plot([],[],color=INK,marker='^',ls='--',label='perception / percDiff')
    D.set(xlabel='model order (60 ms window)',ylabel='parametric ÷ state-space',xticks=[2,4,6,8],
          title='(d) TASK — the collapse replicates exactly.\nNot a property of one dataset')
    D.legend(fontsize=8.5,frameon=False,loc='lower left')
    for a in ax.ravel():
        sns.despine(ax=a); a.grid(axis='y',color=MUT,alpha=0.2,lw=0.6); a.set_axisbelow(True)
    fig.suptitle('Q2 · Does the difference depend on the analysis choices?   '
                 'Only on the WINDOW. Order, inverse and task leave it unchanged.',fontsize=12,y=1.005)
    fig.tight_layout(); fig.savefig(f'{OUT}/story2_parameter_dependence.png',dpi=175,
                                    bbox_inches='tight',facecolor='white'); plt.close(fig)


def story3():
    """Q3 - config F as primary: pairwise vs hypothesis-conditioned, dual stream."""
    import pandas as pd
    # dual-stream framing: which mediator explains auditory -> frontal?
    DORSAL=[('awfa-lh','tpc-lh','pmc-lh'),('awfa-lh','tpc-lh','ifc-lh')]   # via parietal
    VENTRL=[('awfa-lh','ifc-lh','pmc-lh'),('awfa-lh','pmc-lh','ifc-lh')]   # via frontal
    def rows(task,stim):
        _,b,w,st=load_dir(f'{R}/{task}/dSPM/custom/F_triples/win60ms_order6_fs200_pc1/{stim}')
        out=[]
        for (a,m,c) in PRIMARY_TRIPLES:
            k=f'f_ss__{a}__{m}__{c}'; pk=f'f_pair__{a}__{c}'
            if k not in st or pk not in st: continue
            for bd in BANDS:
                bi=b.index(bd)
                un=np.nanmean(st[pk][:,bi],1); cd=np.nanmean(st[k][:,bi],1)
                ok=np.isfinite(un)&np.isfinite(cd)
                if ok.sum()<3: continue
                t=ttest_rel(un[ok],cd[ok])
                out.append(dict(a=a,m=m,c=c,band=bd,un=un[ok].mean(),cd=cd[ok].mean(),
                                M=np.mean(1-cd[ok]/un[ok]),p=t.pvalue,n=ok.sum()))
        import pandas as pd
        d=pd.DataFrame(out); d['p_fdr']=bh_fdr(d['p'].to_numpy()); return d

    fig,axes=plt.subplots(2,3,figsize=(15.0,8.4))
    for r,(task,stim,tl) in enumerate((('overtProd','prodDiff','overt production'),
                                       ('perception','percDiff','perception'))):
        d=rows(task,stim)
        d.to_csv(f'{OUT}/story3_mediation_{task}_{stim}.csv',index=False)
        for c,bd in enumerate(BANDS):
            ax=axes[r][c]; s=d[d.band==bd].copy()
            s['lab']=[f'{sh(x.a)}→{sh(x.c)} | {sh(x.m)}' for x in s.itertuples()]
            s=s.sort_values('un')
            y=np.arange(len(s))
            ax.hlines(y,s.cd,s.un,color=MUT,lw=1.2,zorder=2)
            ax.scatter(s.un,y,s=42,color=UNC,zorder=3,label='pairwise  $F(a\\to c)$')
            ax.scatter(s.cd,y,s=42,color=CON,zorder=4,label='conditional  $F(a\\to c\\,|\\,b)$')
            sig=(s.p_fdr<0.05).to_numpy()
            if sig.any():
                ax.scatter(s.cd.to_numpy()[sig],y[sig],s=110,facecolors='none',
                           edgecolors='#b2182b',lw=1.5,zorder=5)
            ax.set_yticks(y); ax.set_yticklabels(s.lab,fontsize=7.5)
            ax.set_title(f'{tl} — {BL[bd]}',fontsize=10,loc='left',color=INK)
            if r==1: ax.set_xlabel('Granger causality (state-space)',fontsize=9)
            if r==0 and c==0: ax.legend(fontsize=8,frameon=False,loc='lower right')
            ax.set_xlim(left=0)
            sns.despine(ax=ax); ax.grid(axis='x',color=MUT,alpha=0.2,lw=0.6); ax.set_axisbelow(True)
    fig.suptitle('Q3 · Hypothesis-based conditioning (config F, now primary)   ·   '
                 '14 pre-registered triples, 20 subjects, state-space estimator\n'
                 'Grey = influence with no conditioning. Blue = the same influence after conditioning on the '
                 'candidate mediator. A long grey-to-blue line means the mediator explains it away.\n'
                 'Red rings = conditional reliably below pairwise (FDR < 0.05).',fontsize=11,y=1.05)
    fig.tight_layout()
    fig.savefig(f'{OUT}/story3_dual_stream_mediation.png',dpi=175,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    d=rows('perception','percDiff')
    print('perception, theta, sorted by M:')
    print(d[d.band=='theta'].assign(lab=lambda x:[f'{sh(r.a)}->{sh(r.c)}|{sh(r.m)}' for r in x.itertuples()])
          [['lab','un','cd','M','p_fdr']].sort_values('M',ascending=False).head(6).to_string(index=False))
    print('wrote story3')


def main():
    os.makedirs(OUT, exist_ok=True)
    for fn in (story1, story2, story3):
        fn(); print(f'  {fn.__name__} done')
    print(f'figures in {OUT}')


if __name__ == '__main__':
    main()
