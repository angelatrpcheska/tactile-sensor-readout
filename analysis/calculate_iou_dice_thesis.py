#!/usr/bin/env python3
"""Reference-based IoU/Dice for 10-trial tactile_mold_experiment recordings.

Uses the *same built-in geometry and rasterization rule* as the user's former
calculate_iou_dice.py, but reads measured binary masks from measurements.npz
instead of attempting to reconstruct masks from PNG colours.

Examples:
  python3 analysis/calculate_iou_dice_thesis.py \
      --data three_stripe_200g_20261022_170808.zip \
      --out thesis_reference_results

  # PREFERRED if the exact independently defined 66-taxel ERK mask is available:
  python3 analysis/calculate_iou_dice_thesis.py \
      --data three_stripe_200g_20261022_170808.zip \
      --reference dxf_reference_12x32.npy \
      --out thesis_reference_ERK_results

IMPORTANT: Do not tune reference shifts, width, or overlap to maximize IoU on
these recorded trials. They should reflect independent physical geometry and
sensor/mold alignment. The built-in defaults produce 60 taxels, NOT the 66
reference taxels reported for the earlier ERK experiment.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np

ROWS, COLS = 12, 32
PITCH_MM = 2.0
MOLD_WIDTH_MM, MOLD_HEIGHT_MM = 65.0, 26.0
ACTIVE_WIDTH_MM, ACTIVE_HEIGHT_MM = 64.0, 24.0
ORIGIN_X_MM = (MOLD_WIDTH_MM - ACTIVE_WIDTH_MM)/2.0
ORIGIN_Y_MM = (MOLD_HEIGHT_MM - ACTIVE_HEIGHT_MM)/2.0
# Exactly as in calculate_iou_dice(1).py (CAD-derived centre lines, not DXF polygons)
LINES_MM = (
    ((8.3578643763, 6.63603896935), (21.0857864377, 19.36396103070)),
    ((26.1360389694, 6.63603896935), (38.8639610307, 19.36396103070)),
    ((43.9142135624, 6.63603896935), (56.6421356238, 19.36396103070)),
)


def make_geometry_mask(width: float=4.0, overlap: float=0.25, subn: int=40):
    if width <= 0 or not 0 <= overlap <= 1 or subn < 2:
        raise ValueError('Check stripe width (>0), overlap (0..1), supersampling (>=2)')
    sub=(np.arange(subn,dtype=np.float64)+0.5)/subn
    coverage=np.zeros((ROWS,COLS),dtype=np.float64)
    for row in range(ROWS):
        for col in range(COLS):
            x=ORIGIN_X_MM+(col+sub[None,:])*PITCH_MM
            y=ORIGIN_Y_MM+(row+sub[:,None])*PITCH_MM
            xx=np.broadcast_to(x,(subn,subn))
            yy=np.broadcast_to(y,(subn,subn))
            inside=np.zeros((subn,subn),dtype=bool)
            for (x0,y0),(x1,y1) in LINES_MM:
                vx,vy=x1-x0,y1-y0
                length_sq=vx*vx+vy*vy
                length=np.sqrt(length_sq)
                dx,dy=xx-x0,yy-y0
                t=(dx*vx+dy*vy)/length_sq
                perp=np.abs(vx*dy-vy*dx)/length
                inside |= (t>=0)&(t<=1)&(perp<=width/2.0)
            coverage[row,col]=inside.mean()
    return coverage >= overlap, coverage


def load_ref(path: Path):
    if path.suffix.lower()=='.npy':
        a=np.load(path,allow_pickle=False)
    elif path.suffix.lower()=='.csv':
        a=np.loadtxt(path,delimiter=',')
    elif path.suffix.lower()=='.png':
        from PIL import Image
        a=np.asarray(Image.open(path).convert('L'))
        if a.shape==(ROWS,COLS):
            a=a.astype(float)
        elif a.shape[0]%ROWS==0 and a.shape[1]%COLS==0 and a.shape[0]//ROWS==a.shape[1]//COLS:
            bh,bw=a.shape[0]//ROWS,a.shape[1]//COLS
            a=a.reshape(ROWS,bh,COLS,bw).mean(axis=(1,3))
        else:
            raise ValueError('Reference PNG must be clean 12x32 binary image or exact integer enlargement (no axes or labels)')
        a=(a > (0.5 if a.max()<=1 else 127.5)).astype(np.uint8)
    else:
        raise ValueError('Reference must be 12x32 binary .npy, .csv, or clean .png')
    a=np.squeeze(a)
    if a.shape!=(ROWS,COLS):
        raise ValueError(f'Reference shape {a.shape}; expected (12,32)')
    if not np.isfinite(a).all() or not np.isin(a,[0,1]).all() or a.sum()==0:
        raise ValueError('Reference must be a nonempty binary 0/1 mask')
    return a.astype(bool)


def transform(ref, flip_ud=False, flip_lr=False, row_shift=0, col_shift=0):
    if flip_ud: ref=np.flipud(ref)
    if flip_lr: ref=np.fliplr(ref)
    if row_shift==0 and col_shift==0: return ref
    shifted=np.zeros_like(ref)
    sr0,sr1=max(0,-row_shift),min(ROWS,ROWS-row_shift)
    sc0,sc1=max(0,-col_shift),min(COLS,COLS-col_shift)
    tr0,tr1=max(0,row_shift),min(ROWS,ROWS+row_shift)
    tc0,tc1=max(0,col_shift),min(COLS,COLS+col_shift)
    if sr0<sr1 and sc0<sc1:
        shifted[tr0:tr1,tc0:tc1]=ref[sr0:sr1,sc0:sc1]
    return shifted


def read_trials(path: Path):
    if path.is_file() and path.suffix.lower()=='.zip':
        with zipfile.ZipFile(path) as archive:
            files=sorted(n for n in archive.namelist() if n.endswith('/measurements.npz') and '/trial_' in n)
            if not files: raise FileNotFoundError('No trial_*/measurements.npz inside ZIP')
            for name in files:
                with np.load(io.BytesIO(archive.read(name)),allow_pickle=False) as z:
                    yield name.rsplit('/',2)[-2],z['active_mask'].copy(),z['delta'].copy()
    elif path.is_dir():
        files=sorted(path.glob('trial_*/measurements.npz'))
        if not files: raise FileNotFoundError('No trial_*/measurements.npz under directory')
        for f in files:
            with np.load(f,allow_pickle=False) as z:
                yield f.parent.name,z['active_mask'].copy(),z['delta'].copy()
    else:
        raise FileNotFoundError(f'Not a ZIP or extracted session directory: {path}')


def save_images(out: Path, reference, trial_masks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    plt.imsave(out/'reference_mask_used.png',reference.astype(np.uint8),cmap='gray',vmin=0,vmax=1)
    colours=['#f4f4f4','#14815e','#dd7047','#4566c1']
    cm=ListedColormap(colours); norm=BoundaryNorm(np.arange(-.5,4.5),4)
    legend=[Patch(color=c,label=lab) for c,lab in zip(colours,
       ['Neither','TP: both','FP: measured only','FN: reference only'])]
    for name,measured in trial_masks:
        x=np.zeros((ROWS,COLS),dtype=np.uint8)
        x[reference & measured]=1
        x[~reference & measured]=2
        x[reference & ~measured]=3
        fig,ax=plt.subplots(figsize=(11.5,4.9))
        ax.imshow(x,cmap=cm,norm=norm,interpolation='nearest',aspect='equal')
        ax.set_xticks(np.arange(0,COLS,2));ax.set_yticks(np.arange(ROWS))
        ax.set_xticks(np.arange(-.5,COLS,1),minor=True)
        ax.set_yticks(np.arange(-.5,ROWS,1),minor=True)
        ax.grid(which='minor',color='white',linewidth=.4)
        ax.tick_params(which='minor',bottom=False,left=False)
        ax.set_xlabel('Column (0-31)');ax.set_ylabel('Row (0-11)')
        ax.set_title(f'{name}: reference vs measured contact')
        ax.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,-.12),ncol=4,frameon=False,fontsize=9)
        fig.tight_layout()
        fig.savefig(out/'overlays'/f'{name}_overlay.png',dpi=150,bbox_inches='tight')
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data',required=True, type=Path,help='Session ZIP or extracted session folder')
    parser.add_argument('--reference',type=Path,help='Exact independent 12x32 binary ERK reference mask (.npy/.csv/.png); preferred')
    parser.add_argument('--out',type=Path,default=Path('three_stripe_reference_results'))
    parser.add_argument('--threshold',type=float,default=500.,help='Must match saved-mask contact threshold')
    parser.add_argument('--stripe-width',type=float,default=4.,help='Built-in geometric reference width in mm')
    parser.add_argument('--min-overlap',type=float,default=.25,help='Taxel coverage required for reference')
    parser.add_argument('--supersampling',type=int,default=40)
    parser.add_argument('--row-shift',type=int,default=0)
    parser.add_argument('--col-shift',type=int,default=0)
    parser.add_argument('--flip-ud',action='store_true')
    parser.add_argument('--flip-lr',action='store_true')
    parser.add_argument('--expected-reference-taxels',type=int,default=66)
    a=parser.parse_args()
    if a.reference:
        ref=load_ref(a.reference)
        refsource=str(a.reference)
    else:
        ref,_=make_geometry_mask(a.stripe_width,a.min_overlap,a.supersampling)
        refsource=f'original script built-in CAD centre-lines (width={a.stripe_width}mm, overlap={a.min_overlap}, supersampling={a.supersampling})'
    ref=transform(ref,a.flip_ud,a.flip_lr,a.row_shift,a.col_shift)
    if ref.sum()!=a.expected_reference_taxels:
        print(f'[IMPORTANT] Reference contains {ref.sum()} taxels; earlier ERK reported {a.expected_reference_taxels}.')
        print('            Verify whether this is the exact earlier reference BEFORE citing as comparable IoU/Dice.')
    if a.out.exists() and any(a.out.iterdir()):
        raise FileExistsError(f'{a.out} is nonempty: choose a new output path')
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'overlays').mkdir(exist_ok=True)
    np.save(a.out/'reference_mask_used.npy',ref.astype(np.uint8))
    results=[]; trial_masks=[]
    for name,saved_mask,delta in read_trials(a.data):
        if saved_mask.shape!=(ROWS,COLS) or delta.shape!=(ROWS,COLS):
            raise ValueError(f'{name}: unexpected shape')
        measured=saved_mask.astype(bool)
        if not np.isin(saved_mask,[0,1]).all() or not np.isfinite(delta).all():
            raise ValueError(f'{name}: nonbinary mask or nonfinite delta')
        if not np.array_equal(measured,delta>=a.threshold):
            raise ValueError(f'{name}: saved active_mask != (delta >= {a.threshold}); check acquisition threshold')
        tp=int((ref & measured).sum());fp=int((~ref & measured).sum());fn=int((ref & ~measured).sum())
        iou=tp/(tp+fp+fn) if tp+fp+fn else float('nan')
        dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else float('nan')
        results.append({'trial':name,'ref_taxels':int(ref.sum()),'measured_taxels':int(measured.sum()),'TP':tp,'FP':fp,'FN':fn,'IoU':iou,'Dice':dice})
        trial_masks.append((name,measured))
        print(f'{name}: active={int(measured.sum()):2d}, TP={tp:2d}, FP={fp:2d}, FN={fn:2d}, IoU={iou:.4f}, Dice={dice:.4f}')
    if not results: raise RuntimeError('No trials loaded')
    with (a.out/'reference_iou_dice_by_trial.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    xs=np.array([x['IoU'] for x in results]);ys=np.array([x['Dice'] for x in results]);
    summary={'reference_source':refsource,'reference_taxels':int(ref.sum()),'previous_ERK_reference_taxels':a.expected_reference_taxels,
             'reference_matches_ERK_count':bool(ref.sum()==a.expected_reference_taxels),
             'reference_transform':{'flip_ud':a.flip_ud,'flip_lr':a.flip_lr,'row_shift':a.row_shift,'col_shift':a.col_shift},
             'mask_rule':f'contact_delta >= {a.threshold} ADC counts',
             'n_trials':len(results),'mean_IoU':float(xs.mean()),'sample_SD_IoU':float(xs.std(ddof=1)) if len(xs)>1 else None,
             'mean_Dice':float(ys.mean()),'sample_SD_Dice':float(ys.std(ddof=1)) if len(ys)>1 else None,
             'alignment_warning':'Do not optimize geometric registration on these evaluation trials.'}
    (a.out/'reference_iou_dice_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    save_images(a.out,ref,trial_masks)
    print(f'\nMean IoU {xs.mean():.4f} ± {xs.std(ddof=1):.4f}; mean Dice {ys.mean():.4f} ± {ys.std(ddof=1):.4f}')
    print(f'Saved: {a.out.resolve()}')

if __name__=='__main__': main()
