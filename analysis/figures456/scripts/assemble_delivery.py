from pathlib import Path
import json,shutil,html,sys,os
import fitz
ROOT=Path(__file__).resolve().parents[1]
config=ROOT/'package_config.json'
figures=[int(os.environ['PAPER_FIGURE'])] if os.environ.get('PAPER_FIGURE') else ([json.loads(config.read_text())['figure']] if config.exists() else ([int(x) for x in sys.argv[1:]] if len(sys.argv)>1 else [4,5,6]))
def join(dest,files,vertical=False):
    docs=[fitz.open(f) for f in files];sizes=[d[0].rect for d in docs]
    w=max(r.width for r in sizes) if vertical else sum(r.width for r in sizes);h=sum(r.height for r in sizes) if vertical else max(r.height for r in sizes)
    out=fitz.open();page=out.new_page(width=w,height=h);cursor=0
    for d,r in zip(docs,sizes):
        box=fitz.Rect(0,cursor,w,cursor+r.height) if vertical else fitz.Rect(cursor,0,cursor+r.width,h)
        page.show_pdf_page(box,d,0);cursor+=r.height if vertical else r.width
    out.save(dest,garbage=4,deflate=True)
def preview(pdf,png,scale=1.7):
    with fitz.open(pdf) as d:d[0].get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False).save(png)
for number in figures:
    source=ROOT/f'rendered/figure{number}';dest=ROOT if config.exists() else ROOT/f'assembled/figure{number}'
    panels=dest/'panels_pdf';full=dest/'figure_pdf';panels.mkdir(parents=True,exist_ok=True);full.mkdir(exist_ok=True)
    for f in source.glob('*.pdf'):
        name='normalization_mmse_by_lesion_span.pdf' if f.stem=='panel_a_interaction_version2_faceted_tertiles' else f.name
        shutil.copy2(f,panels/name)
    if number==4:
        if not (source/'G2_reference_and_adjusted_effect.pdf').exists():join(panels/'G2_reference_and_adjusted_effect.pdf',[panels/'healthy_reference_G2.pdf',panels/'adjusted_TMS_effect_G2.pdf'])
        layout=[('lesion_volume_pre_post',(0,.02,.59,.25)),('lesion_overlap_maps',(0,.25,.59,.34)),('clinical_subgroup_improvement',(.59,.02,1,.34)),('G2_reference_and_adjusted_effect',(0,.35,.49,.55)),('network_distribution',(.49,.35,.71,.55)),('network_spin_test',(.71,.35,1,.55)),('whole_cortex_normalization',(0,.57,.38,.79)),('normalization_by_distance',(.38,.57,.71,.79)),('individual_distance_slopes',(.71,.57,1,.79)),('lesion_matched_comparison',(0,.81,.29,1)),('normalization_mmse_by_lesion_span',(.29,.81,1,1))]
        letters=[('a',.003,.019),('b',.595,.019),('c',.003,.35),('d',.003,.57),('e',.385,.57),('f',.003,.81),('g',.295,.81)];height=230
    elif number==5:
        for g in ['G1','G2']:join(panels/f'{g}_pre_post_surfaces.pdf',[panels/f'{g}_pre_surface.pdf',panels/f'{g}_post_surface.pdf'])
        for g in ['G1','G2']:join(panels/f'{g}_longitudinal_effect_and_counts.pdf',[panels/f'{g}_longitudinal_effect.pdf',panels/f'{g}_network_ROI_counts.pdf'])
        join(panels/'clinical_association_landscape_and_networks.pdf',[panels/'cortical_clinical_associations.pdf',panels/'clinical_associations_by_network.pdf'],vertical=True)
        layout=[('clinical_score_pre_post',(0,.02,.225,.18)),('gradient_component_variance',(0,.19,.225,.34)),('G1_pre_post_surfaces',(.23,.02,.60,.165)),('G1_longitudinal_effect_and_counts',(.23,.17,.60,.34)),('G2_pre_post_surfaces',(.605,.02,1,.165)),('G2_longitudinal_effect_and_counts',(.605,.17,1,.34)),('network_organization',(0,.36,.225,.65)),('clinical_gradient_association',(.24,.36,.57,.65)),('clinical_association_landscape_and_networks',(.58,.36,1,.65)),('within_and_cross_cohort_beta_maps',(0,.68,.38,1)),('beta_map_spatial_correspondence',(.38,.68,.61,1)),('within_and_cross_cohort_network_spin',(.62,.68,1,1))]
        letters=[('a',.003,.019),('b',.233,.019),('c',.608,.019),('d',.003,.36),('e',.243,.36),('f',.003,.68)];height=297
    else:
        layout=[('primary_cohort_prediction',(0,.025,.50,.95)),('external_cohort_prediction',(.50,.025,1,.95)),('prediction_legend',(.10,.95,.90,1))];letters=[('a',.003,.019),('b',.505,.019)];height=215
    width=(333 if number==5 else 183)/25.4*72;height=height/25.4*72;out=fitz.open();page=out.new_page(width=width,height=height)
    for name,box in layout:
        f=panels/f'{name}.pdf'
        with fitz.open(f) as d:page.show_pdf_page(fitz.Rect(box[0]*width+3,box[1]*height+5,box[2]*width-3,box[3]*height-3),d,0)
    for letter,x,y in letters:page.insert_text((x*width,y*height+3),letter+'.',fontname='hebo',fontsize=11)
    pdf=full/f'Figure_{number}.pdf';out.save(pdf,garbage=4,deflate=True);preview(pdf,pdf.with_suffix('.png'),2)
    cards=[]
    for f in sorted(panels.glob('*.pdf')):
        preview(f,f.with_suffix('.png'),1.2);cards.append(f'<article><a href="panels_pdf/{f.name}"><h2>{html.escape(f.stem)}</h2><img src="panels_pdf/{f.stem}.png"></a></article>')
    (dest/'查看全部子图.html').write_text('<!doctype html><meta charset="utf-8"><title>Figure '+str(number)+'</title><style>body{font:16px Arial,sans-serif;background:#f3f4f6;margin:24px}article{background:white;padding:18px;margin:18px 0}h2{font-size:17px}img{max-width:100%;max-height:850px}a{color:#222;text-decoration:none}</style><h1>Figure '+str(number)+'</h1><p>Individual panels have no panel letters. The composite PDF contains manuscript panel letters.</p>'+''.join(cards),encoding='utf-8')
    print('ASSEMBLED',number,len(list(panels.glob('*.pdf'))),'panels',flush=True)
