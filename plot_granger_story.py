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
    """Q2 - which analysis choice drives the difference?

    Both estimators are plotted as ABSOLUTE GC rather than as a ratio: a
    ratio of group means is undefined in meaning once the numerator crosses
    zero, and it hides magnitude entirely. The failure is quantified as the
    percentage of parametric values below zero, which is bounded and stays
    interpretable exactly where the ratio stopped being so.
    """
    def series(tag,subs,task='overtProd',stim='prodDiff',meth='dSPM',band='theta'):
        """absolute GC per arm, and the % of parametric values that are negative."""
        out=[]
        for sub in subs:
            _,b,_,st=G(tag,sub,task=task,stim=stim,meth=meth); bi=b.index(band)
            d={}
            for meas in ('m1','m2'):
                p_,s_=cat(st,meas+'_par',bi),cat(st,meas+'_ss',bi)
                ok=np.isfinite(p_)&np.isfinite(s_)
                d[meas]=dict(par=p_[ok].mean(),ss=s_[ok].mean(),
                             par_sem=p_[ok].std()/np.sqrt(ok.sum()),
                             ss_sem=s_[ok].std()/np.sqrt(ok.sum()),
                             neg=100*(p_[ok]<0).mean())
            out.append(d)
        return out

    def twoarm(ax,x,rows,meas,xlabel,title,xticks=None,logy=False):
        for arm,col,lab in (('par',PAR,'parametric'),('ss',SS,'state-space')):
            y=[r[meas][arm] for r in rows]; e=[r[meas][arm+'_sem'] for r in rows]
            ax.errorbar(x,y,yerr=e,color=col,lw=2.1,marker='o',ms=5.5,capsize=3,
                        zorder=3,label=lab)
        ax.axhline(0,color='#b2182b',lw=1.4,zorder=2)
        neg=[r[meas]['neg'] for r in rows]
        for xi,n_,yv in zip(x,neg,[r[meas]['par'] for r in rows]):
            if n_>1: ax.annotate(f'{n_:.0f}% <0',(xi,yv),textcoords='offset points',
                                 xytext=(0,-16),ha='center',fontsize=8,color='#b2182b')
        ax.set(xlabel=xlabel,ylabel='Granger causality',title=title)
        if xticks is not None: ax.set_xticks(xticks)
        sns.despine(ax=ax); ax.grid(axis='y',color=MUT,alpha=0.2,lw=0.6); ax.set_axisbelow(True)

    fig,ax=plt.subplots(2,3,figsize=(15.6,8.4))
    wins=[40,60,80,120]; wsub=[f'win{w}ms_order6_fs200_pc1' for w in wins]
    rw=series('B_winsweep',wsub)
    twoarm(ax[0][0],wins,rw,'m2','sliding window (ms), order 6',
           '(a) WINDOW · time-domain GC\nthe parametric mean itself goes negative at 40 ms',xticks=wins)
    # headroom so the legend sits in clear space rather than over the 40 ms point
    y0, y1 = ax[0][0].get_ylim()
    ax[0][0].set_ylim(y0, y1 + 0.34 * (y1 - y0))
    ax[0][0].legend(fontsize=8.5, loc='upper right', frameon=True,
                    framealpha=0.92, facecolor='white', edgecolor='none')
    twoarm(ax[1][0],wins,rw,'m1','sliding window (ms), order 6',
           '(d) WINDOW · conditional spectral GC\nsame sweep, no negatives at all — the arms track',xticks=wins)

    o60=[2,4,6,8]; r60=series('C_ordersweep_win60',[f'win60ms_order{o}_fs200_pc1' for o in o60])
    twoarm(ax[0][1],o60,r60,'m2','model order  ·  60 ms window',
           '(b) ORDER at 60 ms · time-domain GC\nthe gap opens at order 6',xticks=o60)
    o120=[4,8,12,16,20]; r120=series('C_ordersweep_win120',[f'win120ms_order{o}_fs200_pc1' for o in o120])
    twoarm(ax[0][2],o120,r120,'m2','model order  ·  120 ms window',
           '(c) ORDER at 120 ms · same orders, longer window\nthe gap narrows and the failure rate drops, but never to zero',xticks=o120)
    twoarm(ax[1][1],o60,r60,'m1','model order  ·  60 ms window',
           '(e) ORDER at 60 ms · conditional spectral\nunaffected',xticks=o60)

    # (f) inverse and task, as the % of impossible values - one bounded number.
    # Task gets the hue, inverse the shading, so all four bars sit side by side.
    F = ax[1][2]
    w = np.arange(len(wins))
    TASKC = {'overtProd': PAR, 'perception': sns.color_palette('colorblind')[2]}
    combos = [('overtProd', 'prodDiff', 'dSPM'), ('overtProd', 'prodDiff', 'LCMV'),
              ('perception', 'percDiff', 'dSPM'), ('perception', 'percDiff', 'LCMV')]
    bw = 0.2
    for k, (tk, sm, mth) in enumerate(combos):
        rr = series('B_winsweep', wsub, task=tk, stim=sm, meth=mth)
        F.bar(w + (k - 1.5) * bw, [r['m2']['neg'] for r in rr], bw,
              color=TASKC[tk], alpha=(1.0 if mth == 'dSPM' else 0.55),
              edgecolor='white', zorder=3,
              label=f'{tk} · {mth}')
    F.set(xticks=w, xticklabels=[f'{x} ms' for x in wins],
          ylabel='% of parametric values below zero',
          xlabel='sliding window, order 6',
          title='(f) INVERSE and TASK · the failure rate tracks the\nwindow, not the inverse or the task')
    F.legend(fontsize=7.5, frameon=False, ncol=2)
    sns.despine(ax=F)
    F.grid(axis='y', color=MUT, alpha=0.2, lw=0.6)
    F.set_axisbelow(True)

    fig.suptitle('Q2 · Which analysis choice drives the difference?   Both estimators are plotted directly — '
                 'the gap between the lines IS the disagreement.\n'
                 'Red line = zero, which Granger causality cannot cross.',fontsize=11.5,y=1.02)
    fig.tight_layout()
    fig.savefig(f'{OUT}/story2_parameter_dependence.png',dpi=175,bbox_inches='tight',facecolor='white')


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
