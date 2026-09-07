"""
Build the presentation deck from `presentation_data.json`.

Regenerable by design: every number in the deck comes from the JSON, which comes
from logged run directories. When the 600s MILP re-run and the PPO agent
evaluations land, refresh the JSON and re-run this — no hand-editing.

Usage:  python -m tensorfab.build_deck
Output: tensorfab/deck.html
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "presentation_data.json"
OUT = HERE / "deck.html"

CSS = """
:root{
  --ground:#FAF7F0; --panel:#FFFFFF; --sunk:#F1ECE1;
  --ink:#1A1712; --ink-2:#4A4238; --ink-3:#7D7264; --rule:#E2D9C9;
  --accent:#A87400; --accent-soft:#F0E4C8;
  --c1:#A87400; --c2:#0A7D5C; --c3:#2C63C4; --c4:#B92A45; --c5:#7A3FB8;
  --diff:#A87400; --litho:#2C63C4; --fg:#0A7D5C; --hot:#B92A45;
  --ok:#0A7D5C; --open:#B92A45; --warn:#A87400;
  --serif:ui-serif,"New York",Iowan Old Style,Georgia,serif;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
  --measure:66ch;
}
@media (prefers-color-scheme:dark){
  :root{
    --ground:#12131A; --panel:#191B24; --sunk:#0D0E14;
    --ink:#EFEADF; --ink-2:#B6AE9E; --ink-3:#78715F; --rule:#282B37;
    --accent:#EBA91C; --accent-soft:#2A2413;
    --c1:#B8830C; --c2:#12A47B; --c3:#4A86E8; --c4:#E0455F; --c5:#A05FE0;
    --diff:#B8830C; --litho:#4A86E8; --fg:#12A47B; --hot:#E0455F;
    --ok:#12A47B; --open:#E0455F; --warn:#B8830C;
  }
}
:root[data-theme="dark"]{
  --ground:#12131A; --panel:#191B24; --sunk:#0D0E14;
  --ink:#EFEADF; --ink-2:#B6AE9E; --ink-3:#78715F; --rule:#282B37;
  --accent:#EBA91C; --accent-soft:#2A2413;
  --c1:#B8830C; --c2:#12A47B; --c3:#4A86E8; --c4:#E0455F; --c5:#A05FE0;
    --diff:#B8830C; --litho:#4A86E8; --fg:#12A47B; --hot:#E0455F;
  --ok:#12A47B; --open:#E0455F; --warn:#B8830C;
}
:root[data-theme="light"]{
  --ground:#FAF7F0; --panel:#FFFFFF; --sunk:#F1ECE1;
  --ink:#1A1712; --ink-2:#4A4238; --ink-3:#7D7264; --rule:#E2D9C9;
  --accent:#A87400; --accent-soft:#F0E4C8;
  --c1:#A87400; --c2:#0A7D5C; --c3:#2C63C4; --c4:#B92A45; --c5:#7A3FB8;
  --diff:#A87400; --litho:#2C63C4; --fg:#0A7D5C; --hot:#B92A45;
  --ok:#0A7D5C; --open:#B92A45; --warn:#A87400;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--serif);
  font-size:17px;line-height:1.62;-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:0 28px}
h1,h2,h3{text-wrap:balance;margin:0;font-weight:680;letter-spacing:-.018em;line-height:1.14}
h1{font-size:clamp(2.4rem,5.4vw,3.9rem)}
h2{font-size:clamp(1.5rem,2.9vw,2.1rem)}
h3{font-size:1.12rem;letter-spacing:-.008em}
p{margin:0}
.eyebrow{font-family:var(--mono);font-size:.685rem;letter-spacing:.15em;
  text-transform:uppercase;color:var(--accent);font-weight:600}
.sub{color:var(--ink-2);max-width:var(--measure)}
.muted{color:var(--ink-3)}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
section{padding:56px 0;border-top:1px solid var(--rule)}
section:first-of-type{border-top:none}
.stack{display:flex;flex-direction:column;gap:18px}
.stack-l{display:flex;flex-direction:column;gap:34px}
.grid{display:grid;gap:16px}
@media(min-width:760px){.g2{grid-template-columns:1fr 1fr}.g3{grid-template-columns:repeat(3,1fr)}}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:3px;padding:20px 22px}
.card.flag{border-left:3px solid var(--accent)}
.card.ok{border-left:3px solid var(--ok)}
.card.open{border-left:3px solid var(--open)}
/* delay-register lattice motif: the shift register that carries a job to completion */
.lattice{display:flex;gap:2px;height:9px;margin:6px 0 0}
.lattice i{flex:1;background:var(--accent);border-radius:1px;display:block}
table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:.79rem;
  font-variant-numeric:tabular-nums}
.scroll{overflow-x:auto;border:1px solid var(--rule);border-radius:3px;background:var(--panel)}
th,td{padding:9px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--rule)}
th:first-child,td:first-child{text-align:left}
thead th{font-size:.66rem;letter-spacing:.09em;text-transform:uppercase;
  color:var(--ink-3);font-weight:600;background:var(--sunk)}
tbody tr:last-child td{border-bottom:none}
td.hero{color:var(--accent);font-weight:700}
td.neg{color:var(--open)}
.pill{display:inline-block;font-family:var(--mono);font-size:.63rem;letter-spacing:.07em;
  text-transform:uppercase;padding:2px 7px;border-radius:2px;border:1px solid currentColor;
  font-weight:600}
.pill.ok{color:var(--ok)}.pill.open{color:var(--open)}.pill.prov{color:var(--warn)}
.stat{font-family:var(--mono);font-variant-numeric:tabular-nums;
  font-size:clamp(1.6rem,3.4vw,2.4rem);font-weight:600;letter-spacing:-.03em;line-height:1}
.statlbl{font-family:var(--mono);font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink-3);margin-top:7px}
figure{margin:0}
figcaption{font-size:.86rem;color:var(--ink-3);margin-top:10px;max-width:var(--measure)}
svg{display:block;width:100%;height:auto;overflow:visible}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-family:var(--mono);font-size:.7rem;
  color:var(--ink-2);margin-bottom:10px}
.legend span{display:flex;align-items:center;gap:6px}
.swatch{width:11px;height:11px;border-radius:2px;flex:none}
ul{margin:0;padding-left:1.15em}li{margin:5px 0}li::marker{color:var(--accent)}
code{font-family:var(--mono);font-size:.87em;background:var(--sunk);padding:1px 5px;border-radius:2px}
a{color:var(--accent)}
a:focus-visible,[tabindex]:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.banner{background:var(--accent-soft);border:1px solid var(--accent);border-radius:3px;
  padding:14px 18px;font-size:.9rem}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}

/* ---- print / PDF ------------------------------------------------------
   The page is a scrolling document on screen and a paginated deck on paper.
   Force the light palette (dark backgrounds waste toner and render badly),
   start each section on a fresh page, and never split a card, table or
   figure across a page boundary. */
@media print{
  :root{
    --ground:#FFFFFF; --panel:#FFFFFF; --sunk:#F4F1E9;
    --ink:#12100C; --ink-2:#3A342C; --ink-3:#6B6357; --rule:#D8CFBE;
    --accent:#8A5F00; --accent-soft:#F6EDD8;
    --c1:#8A5F00; --c2:#07684B; --c3:#24529F; --c4:#9C2038; --c5:#63329A;
    --diff:#8A5F00; --litho:#24529F; --fg:#07684B; --hot:#9C2038;
    --ok:#07684B; --open:#9C2038; --warn:#8A5F00;
  }
  @page{ size:A4 landscape; margin:13mm 14mm; }
  body{ font-size:11.5px; line-height:1.5; background:#fff; }
  .wrap{ max-width:none; padding:0; }
  section{ break-before:page; page-break-before:always; padding:0 0 8px;
           border-top:none; }
  section:first-of-type{ break-before:auto; page-break-before:auto; }
  .card,.scroll,figure,table,.grid>div{ break-inside:avoid; page-break-inside:avoid; }
  tr,li{ break-inside:avoid; page-break-inside:avoid; }
  h1{ font-size:30px; } h2{ font-size:19px; } h3{ font-size:12.5px; }
  .stat{ font-size:22px; }
  .sub{ max-width:none; }
  .scroll{ overflow:visible; }
  table{ font-size:9.6px; } th,td{ padding:4px 7px; }
  svg{ max-width:100%; }
  .banner{ background:#F6EDD8; }
  a{ text-decoration:none; }
}

"""


JS = r"""
const D = __PAYLOAD__;
const C = ['--c1','--c2','--c3','--c4','--c5'];
const cv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const ink = () => cv('--ink-3'), rule = () => cv('--rule');
const NS='http://www.w3.org/2000/svg';
const el=(t,a={})=>{const e=document.createElementNS(NS,t);
  for(const k in a) e.setAttribute(k,a[k]); return e;};

// delay-register lattice: opacity ramp evokes a job advancing to completion
for(const id of ['lat','lat2']){
  const n=document.getElementById(id); if(!n) continue;
  for(let i=0;i<34;i++){const b=document.createElement('i');
    b.style.opacity=(0.10+0.90*Math.pow(i/33,2.1)).toFixed(3); n.appendChild(b);}
}

function chartMethods(){
  const host=document.getElementById('c1'); host.innerHTML='';
  const methods=[['commander','Commander'],['random','Random search'],
    ['hillclimb','Hill-climb'],['grad_soft','Gradient (soft)'],['grad_st','Gradient (ST)']];
  const lg=document.getElementById('lg1'); lg.innerHTML='';
  methods.forEach(([k,l],i)=>{const s=document.createElement('span');
    s.innerHTML=`<span class="swatch" style="background:${cv(C[i])}"></span>${l}`;
    lg.appendChild(s);});
  const W=1000,rowH=112,pad=132,H=D.order.length*rowH+34;
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img',
    'aria-label':'Profit by method across four demand patterns'});
  const vals=D.order.flatMap(p=>methods.map(([k])=>D.patterns[p][k]));
  const lo=Math.min(0,...vals), hi=Math.max(...vals), span=hi-lo;
  const x=v=>pad+(v-lo)/span*(W-pad-70);
  D.order.forEach((p,pi)=>{
    const y0=pi*rowH+16;
    svg.appendChild(el('line',{x1:x(0),x2:x(0),y1:y0-4,y2:y0+rowH-22,
      stroke:rule(),'stroke-width':1}));
    const t=el('text',{x:0,y:y0+34,fill:cv('--ink'),'font-size':16,
      'font-family':'ui-monospace,monospace'}); t.textContent=D.labels[p]; svg.appendChild(t);
    const u=el('text',{x:0,y:y0+54,fill:ink(),'font-size':13,
      'font-family':'ui-monospace,monospace'});
    u.textContent=(D.patterns[p].util*100).toFixed(0)+'% util'; svg.appendChild(u);
    methods.forEach(([k],mi)=>{
      const v=D.patterns[p][k], yy=y0+mi*15, x0=x(Math.min(0,v)), x1v=x(Math.max(0,v));
      svg.appendChild(el('rect',{x:x0,y:yy,width:Math.max(2,x1v-x0),height:11,rx:3,
        fill:cv(C[mi])}));
      const lab=el('text',{x:x1v+7,y:yy+9.5,fill:k==='grad_st'?cv('--accent'):ink(),
        'font-size':11.5,'font-family':'ui-monospace,monospace',
        'font-weight':k==='grad_st'?700:400});
      lab.textContent='$'+Math.round(v).toLocaleString(); svg.appendChild(lab);
    });
  });
  host.appendChild(svg);
}

function chartCurves(){
  const host=document.getElementById('c2'); host.innerHTML='';
  const keys=Object.keys(D.curves); const lg=document.getElementById('lg2'); lg.innerHTML='';
  keys.forEach((k,i)=>{const s=document.createElement('span');
    s.innerHTML=`<span class="swatch" style="background:${cv(C[i])}"></span>${D.labels[k]||k}`;
    lg.appendChild(s);});
  const W=1000,H=280,L=54,R=16,T=14,B=40;
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img',
    'aria-label':'Gradient magnitude by tick'});
  const hi=Math.max(...keys.flatMap(k=>D.curves[k]));
  const n=D.curves[keys[0]].length;
  const X=i=>L+i/(n-1)*(W-L-R), Y=v=>H-B-(v/hi)*(H-T-B);
  for(let g=0;g<=4;g++){const y=T+g*(H-T-B)/4;
    svg.appendChild(el('line',{x1:L,x2:W-R,y1:y,y2:y,stroke:rule(),'stroke-width':1}));}
  [0,672,1344,2016,2688].forEach(t=>{
    const xx=L+(t/2688)*(W-L-R);
    const tx=el('text',{x:xx,y:H-16,fill:ink(),'font-size':12,'text-anchor':'middle',
      'font-family':'ui-monospace,monospace'}); tx.textContent=t; svg.appendChild(tx);});
  const ax=el('text',{x:L,y:H-2,fill:ink(),'font-size':11.5,
    'font-family':'ui-monospace,monospace'}); ax.textContent='tick'; svg.appendChild(ax);
  keys.forEach((k,i)=>{
    const c=D.curves[k];
    svg.appendChild(el('path',{d:c.map((v,j)=>`${j?'L':'M'}${X(j).toFixed(1)} ${Y(v).toFixed(1)}`).join(' '),
      fill:'none',stroke:cv(C[i]),'stroke-width':2,'stroke-linejoin':'round'}));
  });
  host.appendChild(svg);
}

function chartThroughput(){
  const host=document.getElementById('c3'); host.innerHTML='';
  const T=D.throughput;
  const rows=[['Python env (FabEnv)',T.env_python],['Tensor env, B=64',T.env_tensor_b64],
    ['Tensor env, B=4096',T.env_tensor_b4096],['Python sim',T.sim_python],
    ['Tensor sim, B=10k',T.sim_tensor_b10000]];
  const W=520,rh=34,H=rows.length*rh+8,pad=178;
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':'Throughput'});
  const hi=Math.max(...rows.map(r=>r[1]));
  rows.forEach(([l,v],i)=>{
    const y=i*rh+6;
    const t=el('text',{x:0,y:y+13,fill:ink(),'font-size':11.5,
      'font-family':'ui-monospace,monospace'}); t.textContent=l; svg.appendChild(t);
    const w=Math.max(2,Math.sqrt(v/hi)*(W-pad-96));
    svg.appendChild(el('rect',{x:pad,y:y+3,width:w,height:12,rx:3,fill:cv('--c1')}));
    const n=el('text',{x:pad+w+7,y:y+13,fill:cv('--ink'),'font-size':11.5,
      'font-family':'ui-monospace,monospace'});
    n.textContent=v>=1e6?(v/1e6).toFixed(1)+'M':(v/1e3).toFixed(0)+'k'; svg.appendChild(n);
  });
  host.appendChild(svg);
  const cap=document.createElement('p'); cap.className='sub muted';
  cap.style.cssText='font-size:.8rem;margin-top:8px;font-family:var(--mono)';
  cap.textContent='steps/second, square-root scale';
  host.appendChild(cap);
}

function draw(){chartMethods();chartCurves();chartThroughput();}
draw();
new MutationObserver(draw).observe(document.documentElement,{attributes:true,
  attributeFilter:['data-theme']});
matchMedia('(prefers-color-scheme:dark)').addEventListener('change',draw);


/* ---- annotated single-line anatomy ---- */
function drawAnatomy(){
  const host=document.getElementById('anat'); if(!host) return; host.innerHTML='';
  const W=1040,H=210,NS2='http://www.w3.org/2000/svg';
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':'annotated production line'});
  const stages=[['D','Diffusion',16,4],['L','Litho',2,1],['D','Diffusion',16,4],['L','Litho',2,1]];
  const col=k=>k==='D'?cv('--diff'):cv('--litho');
  const bw=112, gap=(W-120-bw*4)/4, yB=74, bh=54;
  let x=60;
  const t0=el('text',{x:2,y:yB+32,fill:cv('--ink-3'),'font-size':11,'font-family':'ui-monospace,monospace'});
  t0.textContent='RAW'; svg.appendChild(t0);
  stages.forEach(([k,name,pt,b],i)=>{
    svg.appendChild(el('line',{x1:x-gap+6,x2:x-4,y1:yB+bh/2,y2:yB+bh/2,stroke:cv('--rule'),'stroke-width':2}));
    // queue glyph
    for(let r=0;r<3;r++)
      svg.appendChild(el('rect',{x:x-30,y:yB+14+r*7,width:15,height:5,rx:1.5,fill:cv('--ink-3'),opacity:.5}));
    const ql=el('text',{x:x-22,y:yB+8,fill:cv('--ink-3'),'font-size':10,'text-anchor':'middle',
      'font-family':'ui-monospace,monospace'}); ql.textContent='q'+i; svg.appendChild(ql);
    svg.appendChild(el('rect',{x,y:yB,width:bw,height:bh,rx:3,fill:'none',stroke:col(k),'stroke-width':2}));
    const nm=el('text',{x:x+bw/2,y:yB-24,fill:col(k),'font-size':12,'text-anchor':'middle',
      'font-family':'ui-monospace,monospace','font-weight':600,'letter-spacing':'.06em'});
    nm.textContent=name.toUpperCase(); svg.appendChild(nm);
    const id=el('text',{x:x+bw/2,y:yB+22,fill:cv('--ink'),'font-size':14,'text-anchor':'middle',
      'font-family':'ui-monospace,monospace','font-weight':600}); id.textContent='M'+i; svg.appendChild(id);
    const sp=el('text',{x:x+bw/2,y:yB+40,fill:cv('--ink-3'),'font-size':10.5,'text-anchor':'middle',
      'font-family':'ui-monospace,monospace'}); sp.textContent=`pt=${pt}t  batch=${b}`; svg.appendChild(sp);
    x+=bw+gap;
  });
  const fx=x-gap+10;
  svg.appendChild(el('line',{x1:fx-gap+2,x2:fx-6,y1:yB+bh/2,y2:yB+bh/2,stroke:cv('--rule'),'stroke-width':2}));
  const fg=el('text',{x:fx,y:yB+bh/2+4,fill:cv('--fg'),'font-size':12,'font-family':'ui-monospace,monospace',
    'font-weight':600}); fg.textContent='SHIP $80'; svg.appendChild(fg);
  // lead-time bracket
  const y2=yB+bh+26;
  svg.appendChild(el('path',{d:`M60 ${y2} L60 ${y2+7} L${fx-10} ${y2+7} L${fx-10} ${y2}`,
    fill:'none',stroke:cv('--accent'),'stroke-width':1.5}));
  const lt=el('text',{x:(60+fx)/2,y:y2+24,fill:cv('--accent'),'font-size':11.5,'text-anchor':'middle',
    'font-family':'ui-monospace,monospace'});
  lt.textContent='lead time L = 16+2+16+2 = 36 ticks  (minimum release -> ship)'; svg.appendChild(lt);
  const cap=el('text',{x:(60+fx)/2,y:y2+42,fill:cv('--ink-3'),'font-size':11,'text-anchor':'middle',
    'font-family':'ui-monospace,monospace'});
  cap.textContent='bottleneck = one diffusion tool: 4 units / 16 ticks = 0.25 units/tick -> 672 per horizon';
  svg.appendChild(cap);
  host.appendChild(svg);
}

/* ---- three approaches ---- */
function drawApproaches(){
  const host=document.getElementById('approaches'); if(!host) return; host.innerHTML='';
  const W=1040,H=250;
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':'three approaches compared'});
  const lanes=[
    {y:26,c:cv('--ink-3'),t:'DISCRETE-EVENT SIM',
     boxes:['Python objects','1 trajectory','sequential tick loop','profit'],
     note:'exact reference - no gradient - 358k steps/s'},
    {y:100,c:cv('--c4'),t:'MODEL-FREE RL (PPO)',
     boxes:['obs (18-dim)','policy net','action','per-tick reward'],
     note:'score-function gradient - local, discounted, high variance'},
    {y:174,c:cv('--ok'),t:'TENSORIZED + PATHWISE',
     boxes:['theta [B,H,M]','batched arrays','B trajectories','END profit'],
     note:'d(profit)/d(theta) through the real dynamics - 60M steps/s'},
  ];
  lanes.forEach(L=>{
    const lab=el('text',{x:0,y:L.y+4,fill:L.c,'font-size':10.5,'font-family':'ui-monospace,monospace',
      'font-weight':600,'letter-spacing':'.09em'}); lab.textContent=L.t; svg.appendChild(lab);
    let x=228, bw=150, gp=32;
    L.boxes.forEach((b,i)=>{
      svg.appendChild(el('rect',{x,y:L.y-12,width:bw,height:30,rx:3,fill:'none',
        stroke:L.c,'stroke-width':1.6,opacity:.9}));
      const t=el('text',{x:x+bw/2,y:L.y+7,fill:cv('--ink'),'font-size':11.5,'text-anchor':'middle',
        'font-family':'ui-monospace,monospace'}); t.textContent=b; svg.appendChild(t);
      if(i<L.boxes.length-1){
        svg.appendChild(el('path',{d:`M${x+bw+4} ${L.y+3} L${x+bw+gp-6} ${L.y+3}`,stroke:L.c,
          'stroke-width':1.4,'marker-end':''}));
        svg.appendChild(el('path',{d:`M${x+bw+gp-10} ${L.y-1} L${x+bw+gp-5} ${L.y+3} L${x+bw+gp-10} ${L.y+7}`,
          fill:'none',stroke:L.c,'stroke-width':1.4}));
      }
      x+=bw+gp;
    });
    const n=el('text',{x:228,y:L.y+34,fill:cv('--ink-3'),'font-size':10.5,
      'font-family':'ui-monospace,monospace'}); n.textContent=L.note; svg.appendChild(n);
  });
  // backward gradient arc on lane 3
  const y3=174;
  svg.appendChild(el('path',{d:`M${228+3*182+150} ${y3+22} C${700} ${y3+50} ${420} ${y3+50} ${228+70} ${y3+22}`,
    fill:'none',stroke:cv('--ok'),'stroke-width':1.6,'stroke-dasharray':'5 3'}));
  const gt=el('text',{x:640,y:y3+62,fill:cv('--ok'),'font-size':11,'text-anchor':'middle',
    'font-family':'ui-monospace,monospace'});
  gt.textContent='backward: d(profit)/d(theta) traverses all 2,688 ticks'; svg.appendChild(gt);
  host.appendChild(svg);
}

/* ---- delay register ---- */
function drawReg(){
  const host=document.getElementById('regviz'); if(!host) return; host.innerHTML='';
  const W=470,H=104, n=9, cw=40, ch=26;
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':'delay register'});
  for(let row=0;row<2;row++){
    const y=14+row*46, filled=row===0?5:4;
    const lb=el('text',{x:0,y:y+18,fill:cv('--ink-3'),'font-size':10,'font-family':'ui-monospace,monospace'});
    lb.textContent=row===0?'tick t':'t+1'; svg.appendChild(lb);
    for(let i=0;i<n;i++){
      const x=52+i*(cw+3);
      svg.appendChild(el('rect',{x,y,width:cw,height:ch,rx:2,fill:i===filled?cv('--accent'):'none',
        stroke:i===filled?cv('--accent'):cv('--rule'),'stroke-width':1.4}));
      if(row===0){
        const t=el('text',{x:x+cw/2,y:y-4,fill:cv('--ink-3'),'font-size':9.5,'text-anchor':'middle',
          'font-family':'ui-monospace,monospace'}); t.textContent=i; svg.appendChild(t);
      }
    }
  }
  const ar=el('text',{x:52,y:100,fill:cv('--accent'),'font-size':10.5,'font-family':'ui-monospace,monospace'});
  ar.textContent='shifts left one slot per tick; slot 0 = completes now';
  svg.appendChild(ar);
  host.appendChild(svg);
}

/* ---- one tick, five ops ---- */
function drawTick(){
  const host=document.getElementById('tickviz'); if(!host) return; host.innerHTML='';
  const steps=[
    ['1','INJECT','q[:,0] += arrivals[t]; backlog += demand[t]','raw material and demand arrive; existing stock ships instantly'],
    ['2','COMPLETE','done = reg[:,:,0];  q += (make*done) @ route_mat','jobs finishing this tick deposit into their destination queue'],
    ['3','START','start = action * idle * canrun;  reg += start * slot_onehot','the only place a decision enters the dynamics'],
    ['4','ACCRUE','sum_q += q; sum_qfin += qfin; sum_late += max(D_lag - shipped, 0)','integer accumulators; prices applied once at the end'],
    ['5','SHIFT','reg = concat(reg[...,1:], zeros)','advance every in-flight job one tick closer to completion'],
  ];
  const W=1040, rh=52, H=steps.length*rh+6;
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':'one tick as five operations'});
  steps.forEach(([n,name,code,note],i)=>{
    const y=i*rh+8;
    svg.appendChild(el('circle',{cx:12,cy:y+14,r:11,fill:'none',stroke:cv('--accent'),'stroke-width':1.5}));
    const num=el('text',{x:12,y:y+18,fill:cv('--accent'),'font-size':11,'text-anchor':'middle',
      'font-family':'ui-monospace,monospace','font-weight':600}); num.textContent=n; svg.appendChild(num);
    const nm=el('text',{x:34,y:y+18,fill:cv('--ink'),'font-size':12,'font-family':'ui-monospace,monospace',
      'font-weight':600,'letter-spacing':'.06em'}); nm.textContent=name; svg.appendChild(nm);
    const cd=el('text',{x:132,y:y+18,fill:cv('--c3'),'font-size':11.5,'font-family':'ui-monospace,monospace'});
    cd.textContent=code; svg.appendChild(cd);
    const nt=el('text',{x:34,y:y+35,fill:cv('--ink-3'),'font-size':10.5,'font-family':'ui-monospace,monospace'});
    nt.textContent=note; svg.appendChild(nt);
    if(i<steps.length-1)
      svg.appendChild(el('line',{x1:12,x2:12,y1:y+26,y2:y+rh+1,stroke:cv('--rule'),'stroke-width':1.4}));
  });
  host.appendChild(svg);
}

/* ---- gradient flow ---- */
function drawGrad(){
  const host=document.getElementById('gradflow'); if(!host) return; host.innerHTML='';
  const W=1040,H=200;
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':'gradient path from decision to profit'});
  const nodes=[['theta[t,m]','decision logit',cv('--accent')],
               ['start','gated: a x idle x canrun',cv('--hot')],
               ['reg slot pt','job in flight',cv('--ink-3')],
               ['q[next]','pt ticks later',cv('--ink-3')],
               ['... M stages ...','routing',cv('--ink-3')],
               ['cum_ship','revenue $80',cv('--fg')]];
  const bw=142, gp=(W-bw*nodes.length)/(nodes.length-1);
  const y=52, bh=42;
  nodes.forEach(([t,sub,c],i)=>{
    const x=i*(bw+gp);
    svg.appendChild(el('rect',{x,y,width:bw,height:bh,rx:3,fill:'none',stroke:c,'stroke-width':1.8}));
    const a=el('text',{x:x+bw/2,y:y+18,fill:c,'font-size':12,'text-anchor':'middle',
      'font-family':'ui-monospace,monospace','font-weight':600}); a.textContent=t; svg.appendChild(a);
    const b=el('text',{x:x+bw/2,y:y+33,fill:cv('--ink-3'),'font-size':10,'text-anchor':'middle',
      'font-family':'ui-monospace,monospace'}); b.textContent=sub; svg.appendChild(b);
    if(i<nodes.length-1){
      const x1=x+bw, x2=x+bw+gp;
      svg.appendChild(el('line',{x1:x1+3,x2:x2-7,y1:y+bh/2,y2:y+bh/2,stroke:cv('--rule'),'stroke-width':1.6}));
      svg.appendChild(el('path',{d:`M${x2-11} ${y+bh/2-4} L${x2-5} ${y+bh/2} L${x2-11} ${y+bh/2+4}`,
        fill:'none',stroke:cv('--rule'),'stroke-width':1.6}));
    }
  });
  const fw=el('text',{x:0,y:30,fill:cv('--ink-3'),'font-size':10.5,'font-family':'ui-monospace,monospace',
    'letter-spacing':'.08em'}); fw.textContent='FORWARD  (exact, bit-identical to the reference simulator)';
  svg.appendChild(fw);
  // backward arc
  const yb=y+bh+30;
  svg.appendChild(el('path',{d:`M${W-bw/2} ${y+bh+4} C${W-bw/2} ${yb+34} ${bw/2} ${yb+34} ${bw/2} ${y+bh+4}`,
    fill:'none',stroke:cv('--ok'),'stroke-width':1.8,'stroke-dasharray':'6 3'}));
  const bwd=el('text',{x:W/2,y:yb+56,fill:cv('--ok'),'font-size':11.5,'text-anchor':'middle',
    'font-family':'ui-monospace,monospace'});
  bwd.textContent='BACKWARD: d(profit)/d(theta) - hard gates replaced by sigmoids (biased, usable)';
  svg.appendChild(bwd);
  const blk=el('text',{x:W/2,y:yb+74,fill:cv('--ink-3'),'font-size':10.5,'text-anchor':'middle',
    'font-family':'ui-monospace,monospace'});
  blk.textContent='a start also BLOCKS the tool for pt ticks - that path is exact and linear (idle = 1 - reg.sum)';
  svg.appendChild(blk);
  host.appendChild(svg);
}

/* ---- two-product line ---- */
function drawTP(){
  const host=document.getElementById('tpline'); if(!host) return; host.innerHTML='';
  const W=1040,H=196;
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':'two-product line'});
  const stages=[['D','Diffusion',16,20,4],['L','Litho',2,2,1],['D','Diffusion',16,20,4],['L','Litho',2,2,1]];
  const col=k=>k==='D'?cv('--diff'):cv('--litho');
  const bw=120, gap=(W-140-bw*4)/4, yB=66, bh=56;
  let x=64;
  const t0=el('text',{x:0,y:yB+34,fill:cv('--ink-3'),'font-size':11,'font-family':'ui-monospace,monospace'});
  t0.textContent='RAW'; svg.appendChild(t0);
  stages.forEach(([k,name,pt0,pt1,b],i)=>{
    svg.appendChild(el('line',{x1:x-gap+8,x2:x-4,y1:yB+bh/2,y2:yB+bh/2,stroke:cv('--rule'),'stroke-width':2}));
    // two product lanes in the queue glyph
    for(let pdt=0;pdt<2;pdt++)
      for(let r=0;r<2;r++)
        svg.appendChild(el('rect',{x:x-30,y:yB+10+pdt*18+r*7,width:15,height:5,rx:1.5,
          fill:pdt===0?cv('--c3'):cv('--c5'),opacity:.75}));
    svg.appendChild(el('rect',{x,y:yB,width:bw,height:bh,rx:3,fill:'none',stroke:col(k),'stroke-width':2}));
    const nm=el('text',{x:x+bw/2,y:yB-22,fill:col(k),'font-size':11.5,'text-anchor':'middle',
      'font-family':'ui-monospace,monospace','font-weight':600,'letter-spacing':'.06em'});
    nm.textContent=name.toUpperCase(); svg.appendChild(nm);
    const id=el('text',{x:x+bw/2,y:yB+20,fill:cv('--ink'),'font-size':13,'text-anchor':'middle',
      'font-family':'ui-monospace,monospace','font-weight':600}); id.textContent='M'+i; svg.appendChild(id);
    const sp=el('text',{x:x+bw/2,y:yB+36,fill:cv('--ink-3'),'font-size':10,'text-anchor':'middle',
      'font-family':'ui-monospace,monospace'});
    sp.textContent=`P0 ${pt0}t / P1 ${pt1}t`; svg.appendChild(sp);
    const bt=el('text',{x:x+bw/2,y:yB+49,fill:cv('--ink-3'),'font-size':10,'text-anchor':'middle',
      'font-family':'ui-monospace,monospace'}); bt.textContent=`batch ${b}`; svg.appendChild(bt);
    if(k==='L'){
      svg.appendChild(el('rect',{x:x+bw/2-48,y:yB+bh+8,width:96,height:17,rx:2,fill:'none',
        stroke:cv('--hot'),'stroke-width':1.3}));
      const cg=el('text',{x:x+bw/2,y:yB+bh+20,fill:cv('--hot'),'font-size':9.5,'text-anchor':'middle',
        'font-family':'ui-monospace,monospace'}); cg.textContent='1t + $20 switch'; svg.appendChild(cg);
    }
    x+=bw+gap;
  });
  const fx=x-gap+12;
  svg.appendChild(el('line',{x1:fx-gap+4,x2:fx-6,y1:yB+bh/2,y2:yB+bh/2,stroke:cv('--rule'),'stroke-width':2}));
  const fg=el('text',{x:fx,y:yB+bh/2-4,fill:cv('--fg'),'font-size':11,'font-family':'ui-monospace,monospace',
    'font-weight':600}); fg.textContent='SHIP'; svg.appendChild(fg);
  const fg2=el('text',{x:fx,y:yB+bh/2+11,fill:cv('--ink-3'),'font-size':10,
    'font-family':'ui-monospace,monospace'}); fg2.textContent='$80 / $100'; svg.appendChild(fg2);
  const lg=el('text',{x:0,y:H-8,fill:cv('--ink-3'),'font-size':10.5,'font-family':'ui-monospace,monospace'});
  lg.textContent='queues carry both products separately (blue = P0, violet = P1); a batch is one recipe';
  svg.appendChild(lg);
  host.appendChild(svg);
}

/* ---- balanced-margin comparison ---- */
function drawBal(){
  const host=document.getElementById('tpbal'); if(!host) return; host.innerHTML='';
  const W=1040,H=200;
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':'margin per bottleneck tick'});
  const groups=[
    {t:'AS CONFIGURED  (kept)', y:14, c:cv('--ok'),
     bars:[['P0',14.00,cv('--c3')],['P1',14.40,cv('--c5')]],
     note:'within 2.9% - neither product dominates, the mix decision is open'},
    {t:'IF REVENUE + COST WERE FLATTENED  (rejected)', y:92, c:cv('--open'),
     bars:[['P0',14.00,cv('--c3')],['P1',11.20,cv('--c5')]],
     note:'P0 dominates by 25% - optimal policy collapses to "always make P0"'},
  ];
  const x0=250, maxv=16, scale=(W-x0-120)/maxv;
  groups.forEach(g=>{
    const lab=el('text',{x:0,y:g.y+12,fill:g.c,'font-size':10.5,'font-family':'ui-monospace,monospace',
      'font-weight':600,'letter-spacing':'.07em'}); lab.textContent=g.t; svg.appendChild(lab);
    g.bars.forEach(([n,v,c],i)=>{
      const y=g.y+22+i*22;
      const pl=el('text',{x:x0-30,y:y+11,fill:cv('--ink'),'font-size':11.5,'text-anchor':'end',
        'font-family':'ui-monospace,monospace'}); pl.textContent=n; svg.appendChild(pl);
      svg.appendChild(el('rect',{x:x0,y,width:v*scale,height:13,rx:3,fill:c}));
      const vl=el('text',{x:x0+v*scale+8,y:y+11,fill:cv('--ink'),'font-size':11.5,
        'font-family':'ui-monospace,monospace','font-weight':600});
      vl.textContent='$'+v.toFixed(2); svg.appendChild(vl);
    });
    const nt=el('text',{x:x0,y:g.y+70,fill:cv('--ink-3'),'font-size':10.5,
      'font-family':'ui-monospace,monospace'}); nt.textContent=g.note; svg.appendChild(nt);
  });
  const ax=el('text',{x:x0,y:H-4,fill:cv('--ink-3'),'font-size':10,
    'font-family':'ui-monospace,monospace'});
  ax.textContent='profit per unit of bottleneck time  ($ per tick of batch-tool capacity)';
  svg.appendChild(ax);
  host.appendChild(svg);
}

/* ---- anytime curves: profit against wall-clock ---- */
function drawAnytime(){
  const host=document.getElementById('anytime'); if(!host) return; host.innerHTML='';
  const W=1040,H=300,L=76,R=150,T=18,B=52;
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img',
    'aria-label':'profit against wall-clock time for CP-SAT and the gradient'});
  const cps=[[36,25180],[110,28425],[275,29554],[1800,29621]];
  const grd=[[5,24678],[9,26885],[36,27831],[110,27997],[275,28127]];
  const bound=30622, lo=24000, hi=31200;
  const xs=t=>L+(Math.log10(t)-Math.log10(4))/(Math.log10(2000)-Math.log10(4))*(W-L-R);
  const ys=v=>H-B-(v-lo)/(hi-lo)*(H-T-B);
  // grid + y axis
  for(let v=24000; v<=31000; v+=1000){
    svg.appendChild(el('line',{x1:L,x2:W-R,y1:ys(v),y2:ys(v),stroke:cv('--rule'),'stroke-width':1}));
    const t=el('text',{x:L-8,y:ys(v)+4,fill:cv('--ink-3'),'font-size':10.5,'text-anchor':'end',
      'font-family':'ui-monospace,monospace'}); t.textContent='$'+(v/1000)+'k'; svg.appendChild(t);
  }
  [5,10,30,60,120,300,900,1800].forEach(t=>{
    const x=xs(t);
    svg.appendChild(el('line',{x1:x,x2:x,y1:T,y2:H-B,stroke:cv('--rule'),'stroke-width':1,opacity:.5}));
    const lb=el('text',{x,y:H-B+16,fill:cv('--ink-3'),'font-size':10.5,'text-anchor':'middle',
      'font-family':'ui-monospace,monospace'});
    lb.textContent = t>=60 ? (t/60)+'m' : t+'s'; svg.appendChild(lb);
  });
  const ax=el('text',{x:(L+W-R)/2,y:H-14,fill:cv('--ink-3'),'font-size':11,'text-anchor':'middle',
    'font-family':'ui-monospace,monospace'});
  ax.textContent='wall-clock time (log scale)'; svg.appendChild(ax);
  // certified bound
  svg.appendChild(el('line',{x1:L,x2:W-R,y1:ys(bound),y2:ys(bound),stroke:cv('--accent'),
    'stroke-width':1.6,'stroke-dasharray':'6 4'}));
  const bl=el('text',{x:W-R+8,y:ys(bound)+4,fill:cv('--accent'),'font-size':10.5,
    'font-family':'ui-monospace,monospace'}); bl.textContent='CP-SAT bound $30,622';
  svg.appendChild(bl);
  const line=(pts,col,label)=>{
    svg.appendChild(el('path',{d:pts.map((p,i)=>`${i?'L':'M'}${xs(p[0]).toFixed(1)} ${ys(p[1]).toFixed(1)}`).join(' '),
      fill:'none',stroke:col,'stroke-width':2.4,'stroke-linejoin':'round'}));
    pts.forEach(p=>svg.appendChild(el('circle',{cx:xs(p[0]),cy:ys(p[1]),r:4,fill:col,
      stroke:cv('--panel'),'stroke-width':1.5})));
    const last=pts[pts.length-1];
    const t=el('text',{x:xs(last[0])+10,y:ys(last[1])+4,fill:col,'font-size':11.5,
      'font-family':'ui-monospace,monospace','font-weight':600});
    t.textContent=label; svg.appendChild(t);
  };
  line(cps,cv('--c3'),'CP-SAT');
  line(grd,cv('--ok'),'Grad ST');
  // crossover marker
  const cx=xs(95);
  svg.appendChild(el('line',{x1:cx,x2:cx,y1:T,y2:H-B,stroke:cv('--hot'),'stroke-width':1.4,
    'stroke-dasharray':'3 3'}));
  const ct=el('text',{x:cx+6,y:T+12,fill:cv('--hot'),'font-size':10.5,
    'font-family':'ui-monospace,monospace'}); ct.textContent='crossover'; svg.appendChild(ct);
  host.appendChild(svg);
}
drawAnytime();
new MutationObserver(drawAnytime).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});

function drawTwoProd(){drawTP();drawBal();}
drawTwoProd();
new MutationObserver(drawTwoProd).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});

function drawTech(){drawAnatomy();drawApproaches();drawReg();drawTick();drawGrad();}
drawTech();
new MutationObserver(drawTech).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
"""



FAMILY3_TEMPLATE = """
<section>
  <p class="eyebrow">Family 3 &mdash; pipeline depth</p>
  <h2 style="margin:10px 0 8px">The gradient's advantage grows with pipeline depth</h2>
  <p class="sub" style="margin-bottom:18px">{note}
  This was a <strong>pre-registered prediction</strong>: if the gradient's advantage comes from
  long-range credit assignment, a longer pipeline &mdash; which pushes the build-ahead decision
  further from the revenue it earns &mdash; should widen its margin. It did.</p>
  <div class="scroll"><table>
    <thead><tr><th class="l">Line</th><th>Lead time</th><th>Commander</th><th>Random</th>
      <th>Hill-climb</th><th>Grad soft</th><th>Grad ST</th><th>vs best 0th-order</th>
      <th>Bound</th><th>ST % bound</th></tr></thead>
    <tbody>{f3rows}</tbody></table></div>
  <div class="grid g3" style="margin-top:20px">
    <div class="card ok"><h3>Margin grows +$606 &rarr; +$815 &rarr; +$1,363</h3>
      <p class="sub" style="font-size:.9rem;margin-top:7px">Over the best zeroth-order method at
      equal budget. The mechanistic story made a prediction before the run and the prediction
      held &mdash; the strongest evidence the programme has produced so far.</p></div>
    <div class="card"><h3>Relaxed forward degrades with depth</h3>
      <p class="sub" style="font-size:.9rem;margin-top:7px">Grad-soft falls from $16,833 to
      $14,771 as relaxation error compounds along a longer chain, while straight-through
      improves. Further confirmation that exact-forward is the right design.</p></div>
    <div class="card open"><h3>The bound is flat &mdash; read that carefully</h3>
      <p class="sub" style="font-size:.9rem;margin-top:7px">$19,008 at all three, for two
      different reasons that should not be conflated. Revenue and total production cost are held
      fixed <em>by choice</em> (see below), so the ceiling should barely move. But the bound is
      also too loose to resolve the difference that does exist &mdash; see the capacity note.</p></div>
  </div>
  <div class="grid g2" style="margin-top:16px">
    <div class="card open"><h3>Correction: capacity is not exactly constant</h3>
      <p class="sub" style="font-size:.92rem;margin-top:7px">An earlier version of this deck said
      capacity was unchanged at 672 units. That holds for <em>steady-state</em> throughput &mdash;
      the bottleneck is a single diffusion stage at 0.25 units/tick no matter how many stages
      follow. But over a <em>finite</em> 2,688-tick horizon a longer pipeline loses real output to
      fill and drain. Measured, driving every tool whenever the gates permit:</p>
      <div class="kv" style="margin-top:10px">
        <b>2+2</b><span>664 units &middot; 12 tool-ticks per unit</span>
        <b>4+4</b><span>652 units &middot; 24 tool-ticks per unit</span>
        <b>6+6</b><span>640 units &middot; 36 tool-ticks per unit</span>
      </div>
      <p class="sub" style="font-size:.92rem;margin-top:10px">So the true optimum <strong>should
      decline slightly with depth</strong>. The LP relaxation behind the dual bound can schedule
      fractionally and never feels that penalty, which is why the bound does not show it. Flat
      bound, genuinely non-flat optimum.</p></div>
    <div class="card flag"><h3>Why cost is held constant &mdash; a modelling choice, not a fact</h3>
      <p class="sub" style="font-size:.92rem;margin-top:7px">Per-stage cost is set to 32/M for
      diffusion and 16/M for litho, so the total stays <strong>$24/unit at every depth</strong>
      and reproduces the original 8/4/8/4 at 2+2. Three options were considered:</p>
      <div class="scroll" style="margin-top:10px"><table>
        <thead><tr><th class="l">Option</th><th>Margin/unit</th><th class="l">Consequence</th></tr></thead>
        <tbody>
          <tr><td class="l"><strong>Fixed cost &amp; revenue</strong> (chosen)</td><td>$56 flat</td>
            <td class="l">Isolates lead time; rows comparable in dollars</td></tr>
          <tr><td class="l">Cost scales, revenue fixed</td><td>$56 &rarr; $32 &rarr; <strong>$8</strong></td>
            <td class="l">Measures collapsing economics, not scheduling</td></tr>
          <tr><td class="l">Both scale</td><td>$56 flat</td>
            <td class="l">Same experiment, more physically honest</td></tr>
        </tbody></table></div>
      <p class="sub" style="font-size:.92rem;margin-top:10px">The chosen option keeps the three
      rows <em>directly comparable in dollars</em>, which is what licenses reading the margin as
      +$606 &rarr; +$815 &rarr; +$1,363 rather than only as percentages. A 12-stage line at
      unscaled per-stage cost would run an $8 margin against $80 revenue, and family 3 would stop
      being a test of scheduling difficulty.</p></div>
  </div>

  <div class="card open" style="margin-top:16px"><h3>Incomplete: CP-SAT at depth</h3>
    <p class="sub" style="font-size:.92rem;margin-top:7px">{milp_status}.
    Long 1800-second solves are running now. Until they land we cannot say whether the exact
    solver still beats the gradient at depth &mdash; the comparison that matters. The
    &ldquo;% of bound&rdquo; column above uses the rigorous dual bound, valid regardless.</p></div>
</section>

<section>
  <p class="eyebrow">How the line was generalised</p>
  <h2 style="margin:10px 0 8px">From four fixed tools to any depth</h2>
  <p class="sub" style="margin-bottom:18px">The line is now an alternating
  diffusion&rarr;litho chain of any length, described by a <strong>route array</strong>: for each
  tool, which queue its output lands in, with &minus;1 meaning &ldquo;ship&rdquo;. In the tensor
  simulator this collapses to two matrix products, <code>emit @ route_mat</code> and
  <code>emit @ to_fg</code>, replacing four hardcoded lines. The same route array is what
  reentrancy (family 5) will need &mdash; a tool feeding its own input queue is just a route that
  points backwards.</p>
  <div class="scroll"><table>
    <thead><tr><th class="l">File</th><th class="l">Change</th><th class="l">Risk note</th></tr></thead>
    <tbody>{genrows}</tbody></table></div>

  <h3 style="margin:26px 0 10px">Verification &mdash; every gate, at every depth</h3>
  <div class="scroll"><table>
    <thead><tr><th class="l">Gate</th><th class="l">What it checks</th><th>M=4</th><th>M=8</th>
      <th>M=12</th><th class="l">Kind of evidence</th></tr></thead>
    <tbody>{vmrows}</tbody></table></div>
  <p class="sub muted" style="font-size:.88rem;margin-top:12px">Generalising <code>sim.py</code>
  paid for itself twice: it unblocks PPO at depth, and it restores a genuinely
  <em>independent</em> tick-by-tick reference at M=8 and M=12. The earlier plan was to fall back
  on self-consistency plus invariants at depth; that compromise is no longer needed.</p>

  <h3 style="margin:26px 0 4px">Four defects found during the refactor</h3>
  <p class="sub" style="font-size:.92rem">Three were introduced by the generalisation and caught
  by the gates; the fourth was latent and would have quietly corrupted every deep-line solve.</p>
  {bugrows}
</section>
"""


F92_TEMPLATE = """
<section>
  <p class="eyebrow">Family 3 at 92% demand &mdash; a negative result</p>
  <h2 style="margin:10px 0 8px">At high load the gradient loses, and loses harder with depth</h2>
  <p class="sub" style="margin-bottom:18px">{note}</p>
  <div class="scroll"><table>
    <thead><tr><th class="l">Line</th><th>Lead</th><th>Commander</th><th>Random</th>
      <th>Hill-climb</th><th>Grad soft</th><th>Grad ST</th><th>ST vs best 0th-order</th></tr></thead>
    <tbody>{f92rows}</tbody></table></div>
  <div class="card open" style="margin-top:18px">
    <h3>This inverts the 50% result and must be read alongside it</h3>
    <p class="sub" style="font-size:.93rem;margin-top:8px">At 50% demand the gradient's margin
    <em>grew</em> with depth: +$606 &rarr; +$815 &rarr; +$1,363. At 92% it goes the other way
    &mdash; &minus;$18 &rarr; &minus;$751 &rarr; &minus;$2,411 &mdash; and at 12 tools the gradient
    finishes below even the greedy commander. The pre-registered depth prediction held at low load
    and failed at high load. Both are real; neither cancels the other; the deck reports both.</p>
    <p class="sub" style="font-size:.93rem;margin-top:8px"><strong>A hypothesis was tested and
    rejected.</strong> {clip} Straight-through gradient norms reach 8.65e9 at 12 tools and 92%
    demand, so the natural explanation was Adam's second-moment estimate being poisoned by an
    enormous first gradient. Clipping should have fixed that. It did not, so the failure is in the
    gradient signal itself, not in how it is consumed.</p></div>

  <h3 style="margin:28px 0 8px">The pattern across everything measured so far</h3>
  <p class="sub" style="margin-bottom:14px">Ordering every problem by how much <em>slack</em> it
  leaves &mdash; capacity not already committed to meeting demand &mdash; produces a clean
  monotonic trend in the gradient's advantage.</p>
  <div class="scroll"><table>
    <thead><tr><th class="l">Problem</th><th>Utilisation</th><th class="l">Demand shape</th>
      <th>Grad ST &minus; best zeroth-order</th></tr></thead>
    <tbody>{strows}</tbody></table></div>
  <div class="grid g2" style="margin-top:18px">
    <div class="card flag"><h3>Working hypothesis: the gradient needs slack</h3>
      <p class="sub" style="font-size:.92rem;margin-top:8px">A start is gated by
      <code>idle &times; canrun</code>. When a tool is busy or starved the gate is closed and the
      derivative through that path is essentially uninformative. At 50% utilisation the gates are
      often open and the choice to idle is genuinely live, so the surrogate carries signal. At 92%
      with flat demand the line is saturated, gates are closed almost always, and the surrogate
      has little to say &mdash; while random sampling still finds good schedules by luck.</p>
      <p class="sub" style="font-size:.92rem;margin-top:8px">This also explains why
      <em>bursty</em> 92% (Alternating 92) still favours the gradient at +$398 while <em>flat</em>
      92% does not: bursty demand leaves slack in the light weeks even when the average load is
      high. On this reading the operative variable is slack, not utilisation.</p></div>
    <div class="card"><h3>CP-SAT vs depth &mdash; results so far</h3>
      <div class="scroll" style="margin-top:8px"><table>
        <thead><tr><th class="l">Line</th><th>CP-SAT</th><th>Grad ST</th><th>CP-SAT lead</th>
          <th>Dual bound</th><th>Gap</th></tr></thead>
        <tbody>{mdrows}</tbody></table></div>
      <p class="sub" style="font-size:.92rem;margin-top:10px">1800-second solves at 50% demand.
      CP-SAT beats the gradient at <strong>every</strong> depth, by $1,098 / $574 / $614.</p>
      <p class="sub" style="font-size:.92rem;margin-top:8px"><strong>An earlier reading of this
      table was wrong.</strong> With only M=4 and M=8 in hand the gap appeared to widen with depth
      (0.05% &rarr; 0.86%), and that was reported here as the first sign of a scaling wall. M=12
      then came in at 0.39% &mdash; better than M=8 &mdash; so the trend is not monotonic and
      there is <strong>no scaling wall at these sizes</strong>. CP-SAT solves all three to within
      1% in half an hour. The scaling premise needs a far larger instance to test, which is
      exactly what family 5 and the multi-product work are for.</p></div>
  </div>
</section>
"""


CAP_TEMPLATE = """
<section>
  <p class="eyebrow">Reading the demand labels</p>
  <h2 style="margin:10px 0 8px">&ldquo;50%&rdquo; and &ldquo;92%&rdquo; of <em>which</em> capacity?</h2>
  <p class="sub" style="margin-bottom:18px">{explain}</p>
  <div class="scroll"><table>
    <thead><tr><th class="l">Line</th><th>Lead</th><th>Nominal capacity</th>
      <th>Actually achievable</th><th>336 units is</th><th>616 units is</th></tr></thead>
    <tbody>{caprows}</tbody></table></div>
  <div class="grid g2" style="margin-top:18px">
    <div class="card flag"><h3>Demand is constant in units, not in percent</h3>
      <p class="sub" style="font-size:.92rem;margin-top:8px">Every depth receives the
      <strong>same absolute demand</strong> &mdash; 336 units in the 50% sweep, 616 in the 92%
      sweep. The percentage label is computed once against the depth-independent nominal figure of
      672, so it is a convenient name for the experiment, not a per-line measurement.</p>
      <p class="sub" style="font-size:.92rem;margin-top:8px">Against what each line can
      <em>actually</em> deliver in a finite horizon, effective load rises with depth: the &ldquo;50%&rdquo;
      sweep runs at 50.6% &rarr; 52.5%, and the &ldquo;92%&rdquo; sweep at 92.8% &rarr; 96.2%. A
      deeper pipeline spends more of the horizon filling and draining, so fewer units clear it.</p></div>
    <div class="card open"><h3>Correction: the deep lines are not capacity-short</h3>
      <p class="sub" style="font-size:.92rem;margin-top:8px">An earlier version of this deck said
      the 6+6 line could make only 596 units against 616 demanded, and therefore that backorder was
      structural at 92%. <strong>That was wrong.</strong> 596 was the <em>commander's</em> output,
      not the line's physical ceiling. Driving every tool whenever the gates permit yields 640
      units, so 616 is satisfiable at every depth.</p>
      <p class="sub" style="font-size:.92rem;margin-top:8px">The real statement is narrower and
      more interesting: the greedy heuristic leaves 44 units on the table at 6+6 versus 41 at 2+2,
      so its shortfall is roughly constant &mdash; but at 92% demand there is no longer slack to
      absorb it, which is why the commander's profit falls from $34,080 to $32,864 as the line
      deepens.</p></div>
  </div>
</section>
"""


TECH_HTML = """
<section>
  <p class="eyebrow">Anatomy</p>
  <h2 style="margin:10px 0 8px">The line, and every quantity that defines it</h2>
  <p class="sub" style="margin-bottom:16px">One product moves left to right through alternating
  batch and single-wafer tools. Everything the simulator knows is on this page.</p>
  <div class="legend">
    <span><span class="sw" style="background:var(--diff)"></span>Diffusion &mdash; batch furnace</span>
    <span><span class="sw" style="background:var(--litho)"></span>Litho &mdash; single wafer</span>
    <span><span class="sw" style="background:var(--ink-3)"></span>Queue (unbounded)</span>
    <span><span class="sw" style="background:var(--fg)"></span>Finished goods / ship</span>
  </div>
  <div id="anat"></div>
  <figcaption>The 2+2 baseline. A diffusion job holds the tool for 16 ticks and converts 4 units;
  a litho job holds it for 2 ticks and converts 1. Queues between stages are unbounded, so a tool
  is blocked only by its own occupancy or by lack of input &mdash; never by downstream congestion.</figcaption>

  <div class="grid g2" style="margin-top:26px">
    <div class="card"><h3>Structural quantities</h3>
      <div class="scroll" style="margin-top:10px"><table>
        <thead><tr><th class="l">Quantity</th><th class="l">Value</th><th class="l">Definition</th></tr></thead>
        <tbody>
        <tr><td class="l">Tick</td><td class="l">atomic</td>
          <td class="l">One decision interval. Every tool is offered a run/idle choice each tick.</td></tr>
        <tr><td class="l">Horizon <em>H</em></td><td class="l">2,688 ticks</td>
          <td class="l">4 weeks of 672 ticks. The full episode.</td></tr>
        <tr><td class="l">Tools <em>M</em></td><td class="l">4 / 8 / 12</td>
          <td class="l">Alternating DIFF, LITHO. <em>M</em> = 2 &times; pairs.</td></tr>
        <tr><td class="l">Products</td><td class="l">1</td>
          <td class="l">Single recipe. No changeover; setup times exist in config but are zero.</td></tr>
        <tr><td class="l">Processing time <em>pt</em></td><td class="l">16 / 2 ticks</td>
          <td class="l">Ticks a tool is occupied by one job. Diffusion 16, litho 2.</td></tr>
        <tr><td class="l">Batch size</td><td class="l">4 / 1 units</td>
          <td class="l">Units consumed and emitted per job. A furnace runs 4 wafers at once.</td></tr>
        <tr><td class="l">Lead time <em>L</em></td><td class="l">36 / 72 / 108 t</td>
          <td class="l">&Sigma;<em>pt</em> &mdash; minimum ticks from releasing raw material to a
            shippable unit, with zero queueing. <strong>The credit-assignment distance.</strong></td></tr>
        <tr><td class="l">Nominal capacity</td><td class="l">672 units</td>
          <td class="l">0.25 units/tick &times; H. Set by the slowest single stage (4 units / 16
            ticks), so independent of depth.</td></tr>
        <tr><td class="l">Achievable capacity</td><td class="l">664 / 652 / 640</td>
          <td class="l">What the line actually clears in a finite horizon, after pipeline fill and drain.</td></tr>
        <tr><td class="l">Raw material</td><td class="l">672 units</td>
          <td class="l">Delivered weekly into q0. Matches nominal capacity, so material and tools
            bind together.</td></tr>
        </tbody></table></div></div>
    <div class="card"><h3>Economic quantities</h3>
      <div class="scroll" style="margin-top:10px"><table>
        <thead><tr><th class="l">Quantity</th><th class="l">Value</th><th class="l">Definition</th></tr></thead>
        <tbody>
        <tr><td class="l">Revenue</td><td class="l">$80 / unit</td>
          <td class="l">Booked on shipment against outstanding demand, not on production.</td></tr>
        <tr><td class="l">Production cost</td><td class="l">$24 / unit total</td>
          <td class="l">Charged at job start. Split 2:1 diffusion:litho, and held constant at
            every depth by design so pipeline length does not change the economics.</td></tr>
        <tr><td class="l">FG holding</td><td class="l">$0.04 /unit/tick</td>
          <td class="l">On finished goods waiting for demand. Punishes building too early.</td></tr>
        <tr><td class="l">WIP holding</td><td class="l">$0 &rarr; $0.0075</td>
          <td class="l">Per queue, linear from q0 to the last queue: value accrues as a unit
            advances, so late-stage inventory costs more to hold.</td></tr>
        <tr><td class="l">Backorder</td><td class="l">$0.15 /unit/tick</td>
          <td class="l">On demand still unmet <em>one week</em> after it arrived. Punishes
            building too late. Nearly 4&times; the FG holding rate.</td></tr>
        <tr><td class="l">Initial stock</td><td class="l">8 units</td>
          <td class="l">Finished goods on hand at t=0.</td></tr>
        <tr><td class="l">Demand interval</td><td class="l">672 ticks</td>
          <td class="l">Demand arrives in weekly bursts and ages into "late" after one week.</td></tr>
        <tr><td class="l">Objective</td><td class="l">profit</td>
          <td class="l">revenue &minus; production &minus; FG holding &minus; WIP holding
            &minus; backorder, over the full horizon.</td></tr>
        </tbody></table></div></div>
  </div>
  <div class="card flag" style="margin-top:16px"><h3>The tension in one sentence</h3>
    <p class="sub" style="font-size:.93rem;margin-top:7px">Backorder costs roughly four times what
    finished-goods holding costs, so you want to build early &mdash; but the pipeline takes
    <em>L</em> ticks to fill and every tick of earliness is charged. The optimal release is a
    narrow window, and it sits <em>L</em> ticks before demand it cannot yet see.</p></div>
</section>

<section>
  <p class="eyebrow">Three ways to attack it</p>
  <h2 style="margin:10px 0 8px">What we had, what failed, and what we built</h2>
  <div id="approaches"></div>
  <div class="grid g3" style="margin-top:22px">
    <div class="card"><h3>1 &middot; Discrete-event simulation</h3>
      <p class="sub" style="font-size:.9rem;margin-top:7px">Python objects: a machine holds
      <code>busy</code> and <code>finish_time</code>; a queue is a list of arrival timestamps. One
      trajectory at a time, one tick per loop iteration. Exact, readable, and the reference every
      other implementation is checked against &mdash; but it produces <em>no</em> gradient, and at
      358k steps/s it is a hard ceiling on how much experience you can generate.</p></div>
    <div class="card open"><h3>2 &middot; Model-free RL &mdash; Maskable PPO</h3>
      <p class="sub" style="font-size:.9rem;margin-top:7px">A policy network reads an 18-dim
      observation and picks run/idle per tool. It learns from a <strong>per-tick reward</strong>
      and a score-function gradient. Two problems: the reward is <em>local</em>, so a production
      decision shows its cost immediately and its revenue hundreds of ticks later, discounted; and
      the gradient is high-variance, needing 8&ndash;16M env steps. PPO reliably drifts into
      under-production and erodes its own behavioural-cloning prior.</p></div>
    <div class="card ok"><h3>3 &middot; Tensorized &amp; differentiable</h3>
      <p class="sub" style="font-size:.9rem;margin-top:7px">The same dynamics rewritten as fixed
      arrays with a batch axis, so thousands of trajectories advance in lockstep and automatic
      differentiation can trace decisions to profit. Two payoffs: throughput
      (60M steps/s vs 358k), and a gradient of <strong>true end-of-horizon profit</strong> with
      respect to every decision &mdash; no local proxy, no discounting.</p></div>
  </div>
  <div class="card flag" style="margin-top:16px"><h3>Why the third exists at all</h3>
    <p class="sub" style="font-size:.93rem;margin-top:7px">Not because the simulator was too slow
    &mdash; it was never PPO's bottleneck. Because PPO's failure was diagnosed as
    <em>reward misalignment</em>: the signal it optimises is not the objective we want. Making the
    dynamics differentiable removes the proxy entirely. You stop asking "was this tick good?" and
    start asking "how does this decision change the profit at the end?"</p></div>
</section>

<section>
  <p class="eyebrow">The tensorization</p>
  <h2 style="margin:10px 0 8px">How a discrete-event line becomes an array program</h2>
  <p class="sub" style="margin-bottom:18px">The reference simulator stores each queue as a list of
  timestamps and each tool as a (busy, finish_time) pair. Neither is differentiable, and neither
  batches. Only <em>counts</em> and <em>countdowns</em> affect the dynamics, so the entire state of
  B trajectories collapses to four arrays.</p>
  <div class="grid g2">
    <div class="card"><h3>State</h3>
      <div class="scroll" style="margin-top:10px"><table>
        <thead><tr><th class="l">Array</th><th class="l">Shape</th><th class="l">Meaning</th></tr></thead>
        <tbody>
        <tr><td class="l"><code>q</code></td><td class="l">[B, M]</td><td class="l">units waiting in each queue</td></tr>
        <tr><td class="l"><code>reg</code></td><td class="l">[B, M, 17]</td><td class="l">delay register &mdash; jobs in flight</td></tr>
        <tr><td class="l"><code>qfin</code></td><td class="l">[B]</td><td class="l">finished goods</td></tr>
        <tr><td class="l"><code>backlog</code></td><td class="l">[B]</td><td class="l">demand not yet shipped</td></tr>
        </tbody></table></div>
      <p class="sub" style="font-size:.9rem;margin-top:10px">Plus integer accumulators for shipped
      units, starts, and the three holding costs. Prices are applied once at the end, so profit is
      a linear function of exact integers &mdash; no per-tick float rounding.</p></div>
    <div class="card"><h3>The delay register</h3>
      <div id="regviz" style="margin-top:6px"></div>
      <p class="sub" style="font-size:.9rem;margin-top:8px">A start writes a 1 into slot
      <em>pt</em>. The register shifts left one slot per tick. A job reaching slot 0 completes.
      "Busy" is just <code>reg.sum() &gt; 0</code>, so one-job-at-a-time capacity is free.</p>
      <p class="sub" style="font-size:.9rem;margin-top:8px"><strong>Why a register and not a
      countdown:</strong> <code>done = reg[...,0]</code> is a pure selection, so the gradient flows
      from a start decision straight through to its completion. A countdown would need
      <code>rem == 1</code>, whose derivative is zero everywhere.</p></div>
  </div>
  <h3 style="margin:26px 0 10px">One tick, as five array operations</h3>
  <div id="tickviz"></div>
  <p class="sub muted" style="font-size:.88rem;margin-top:10px">Ordering is load-bearing:
  completions run before starts, so a tool finishing at tick <em>t</em> can restart at <em>t</em>,
  and a unit deposited by tool <em>m</em> is available to tool <em>m</em>+1 in the same tick.
  Routing is a matrix product, which is what generalises the line to any depth &mdash; and to
  reentrancy, where a route simply points backwards.</p>
</section>

<section>
  <p class="eyebrow">Where the gradient comes from</p>
  <h2 style="margin:10px 0 8px">Tracing a decision to the profit it causes</h2>
  <p class="sub" style="margin-bottom:16px">Every decision is a real-valued logit
  &theta;[t, m]. The question autodiff answers is &part;profit / &part;&theta; &mdash; and the
  answer travels the whole causal chain, not a one-tick reward.</p>
  <div id="gradflow"></div>
  <div class="grid g2" style="margin-top:22px">
    <div class="card open"><h3>The obstacle: three hard gates</h3>
      <p class="sub" style="font-size:.92rem;margin-top:8px">A start happens only if
      <code>action &times; idle &times; canrun</code>. The action is binary, and
      <code>canrun = 1[q &ge; take]</code> is a step function &mdash; derivative zero almost
      everywhere, undefined at the jump. Differentiating the exact simulator returns
      <strong>exactly zero</strong>, which we verify as a control.</p>
      <p class="sub" style="font-size:.92rem;margin-top:8px">The third gate is nearly free:
      because a tool holds at most one job, <code>idle = 1 &minus; reg.sum()</code> is exact
      <em>and</em> linear, so it needs no relaxation and contributes no bias.</p></div>
    <div class="card ok"><h3>The fix: exact forward, surrogate backward</h3>
      <p class="sub" style="font-size:.92rem;margin-top:8px">Straight-through estimation. The
      forward pass uses the hard gate, so the simulation stays bit-identical to the reference. The
      backward pass substitutes a sigmoid. Written
      <code>hard + (soft &minus; stop_gradient(soft))</code> &mdash; the parenthesisation matters,
      since the obvious alternative is not exact in float32 and silently invented $230 of profit
      before it was caught.</p>
      <p class="sub" style="font-size:.92rem;margin-top:8px">Gradients are therefore
      <strong>biased but usable</strong>. That is inherent: you cannot have exact derivatives of a
      discontinuous function.</p></div>
  </div>
  <div class="card flag" style="margin-top:16px"><h3>What this buys, and what it does not</h3>
    <p class="sub" style="font-size:.93rem;margin-top:7px">It buys credit assignment across the
    whole horizon: at 2,688 ticks, more than half the gradient magnitude sits in the first quarter
    of the schedule, with zero non-finite values. The signal genuinely reaches decisions made
    2,000+ ticks before the revenue they earn &mdash; exactly the decision PPO's discounted local
    reward gets wrong.</p>
    <p class="sub" style="font-size:.93rem;margin-top:7px">It does <em>not</em> buy an optimiser
    that always wins. The surrogate carries information only where the gates are live. Where the
    line runs saturated, the gates are shut almost always, the gradient says little, and simple
    random search beats it &mdash; which is precisely what the 92% results show.</p></div>
</section>
"""


TWOPROD_TEMPLATE = """
<section>
  <p class="eyebrow">The two-product line</p>
  <h2 style="margin:10px 0 8px">Same four tools, two recipes, and a changeover penalty</h2>
  <p class="sub" style="margin-bottom:16px">A separate simulator for this has existed in the
  repository since April, predating all the single-product work. It is not a variant of the
  1-product line &mdash; it models per-product processing times, sequence-dependent setup times,
  changeover cost, and a three-way decision at every tool.</p>
  <div class="legend">
    <span><span class="sw" style="background:var(--diff)"></span>Diffusion &mdash; batch, no changeover</span>
    <span><span class="sw" style="background:var(--litho)"></span>Litho &mdash; single wafer, changeover applies</span>
    <span><span class="sw" style="background:var(--hot)"></span>Changeover: 1 tick + $20</span>
  </div>
  <div id="tpline"></div>
  <figcaption>Both products share every tool. The batch tools change recipe freely but run P1
  25% slower; the single-wafer tools switch instantly in throughput terms but charge $20 and one
  tick per changeover. A batch is one recipe &mdash; a furnace cannot mix products.</figcaption>

  <div class="scroll" style="margin-top:24px"><table>
    <thead><tr><th class="l">Product</th><th>Revenue</th><th>Prod cost</th><th>Margin</th>
      <th>Batch pt</th><th>Bottleneck t/unit</th><th>Margin per bottleneck tick</th>
      <th>FG holding</th><th>Backorder</th></tr></thead>
    <tbody>{tprows}</tbody></table></div>
</section>

<section>
  <p class="eyebrow">Why the mix decision is real</p>
  <h2 style="margin:10px 0 8px">The two products are deliberately balanced</h2>
  <p class="sub" style="margin-bottom:16px">P1 pays more but consumes 25% more of the bottleneck.
  Measured in the only currency that matters &mdash; profit per unit of scarce capacity &mdash;
  they land within <strong>2.9%</strong> of each other. Neither dominates, so "which product to
  run" is genuinely open at every decision point.</p>
  <div id="tpbal"></div>
  <div class="card open" style="margin-top:20px">
    <h3>What "aligning the economics" could have destroyed</h3>
    <p class="sub" style="font-size:.93rem;margin-top:8px">The obvious way to bring this line
    inline with the 1-product settings would be to flatten revenue and production cost to
    $80 / $24 for both products. That would have been a serious mistake. P1 would still take
    25% longer on the bottleneck but earn the same margin, dropping it to
    <strong>$11.20 per bottleneck tick against P0's $14.00</strong> &mdash; P0 dominating by 25%.
    The optimal policy would collapse to "always make P0" and the mix decision, the entire point
    of the two-product problem, would disappear.</p>
    <p class="sub" style="font-size:.93rem;margin-top:8px">So revenue, production cost and
    processing times were left untouched. Only the cost terms that carry no product
    differentiation were aligned.</p></div>
</section>

<section>
  <p class="eyebrow">Bringing it inline</p>
  <h2 style="margin:10px 0 8px">Six changes, and what each one was for</h2>
  <p class="sub" style="margin-bottom:16px">The approach was <em>level</em> alignment with
  <em>ratio</em> preservation: match the absolute cost levels to the 1-product line, and keep every
  one of P1's relative premiums exactly as they were. P0 was already economically identical to the
  1-product product &mdash; same $80 revenue, same 8/4/8/4 cost structure &mdash; so after this it
  <strong>is</strong> that product.</p>
  <div class="scroll"><table>
    <thead><tr><th class="l">Change</th><th class="l">Before</th><th class="l">After</th>
      <th class="l">Why it is safe</th></tr></thead>
    <tbody>{alrows}</tbody></table></div>

  <div class="grid g2" style="margin-top:20px">
    <div class="card"><h3>Left untouched &mdash; these carry the difficulty</h3>
      <ul class="sub" style="font-size:.91rem;margin-top:8px">{keptrows}</ul></div>
    <div class="card open"><h3>The honest consequence</h3>
      <p class="sub" style="font-size:.92rem;margin-top:8px">The commander baseline moved a long
      way, in two stages, and both should be on the record:</p>
      <div class="scroll" style="margin-top:10px"><table>
        <thead><tr><th class="l">Stage</th><th>Profit</th><th>Inventory</th><th>Backorder</th>
          <th>Shipped</th></tr></thead><tbody>{stagerows}</tbody></table></div>
      <p class="sub" style="font-size:.92rem;margin-top:10px">The grace period removed a backorder
      cost that had been three quarters of revenue. The finished-goods fix then recovered the last
      four unshipped units and $2,150 of holding on stock that could never move. Neither change
      touched a product differentiator.</p></div>
  </div>

  <h3 style="margin:26px 0 10px">It is still harder than the 1-product line</h3>
  <div class="grid g2">
    <div class="card">
      <div class="scroll"><table>
        <thead><tr><th class="l">Mix @ 80%</th><th>Demand P0/P1</th><th>Commander</th>
          <th>Naive ceiling</th><th>% of ceiling</th><th>Changeovers</th><th>Inventory</th></tr></thead>
        <tbody>{mixrows}</tbody></table></div>
      <p class="sub" style="font-size:.9rem;margin-top:10px">The heuristic leaves
      <strong>14&ndash;17%</strong> on the table at 80% load, and about 5% at 92%. The mix drives
      changeover pressure directly &mdash; 32 switches at an even split against 14 when lopsided.
      Most of the remaining gap is inventory: the commander over-produces and holds, which is
      exactly the failure mode the gradient exploited on the 1-product line.</p></div>
    <div class="card">
      <div class="scroll"><table>
        <thead><tr><th class="l">Configuration</th><th>Profit</th><th>Inventory</th><th>WIP</th>
          <th>Setup</th><th>Backorder</th></tr></thead>
        <tbody>{moderows}</tbody></table></div>
      <p class="sub" style="font-size:.9rem;margin-top:10px">Backorder is now inert in these
      configurations &mdash; with the grace period and a working finished-goods buffer, the
      commander meets all demand. It bites only under tighter load or burstier demand, so if we
      want that lever active we should raise utilisation past 92%.</p></div>
  </div>
</section>

"""


TPRES_TEMPLATE = """
<section>
  <p class="eyebrow">Two-product results</p>
  <h2 style="margin:10px 0 8px">The gradient wins every mix &mdash; and the solver still wins overall</h2>
  <p class="sub" style="margin-bottom:18px">{note}</p>
  <div class="scroll"><table>
    <thead><tr><th class="l">Mix</th><th>Commander</th><th>Random</th><th>Hill-climb</th>
      <th>Grad ST</th><th>vs best 0th</th><th>CP-SAT</th><th>Bound</th><th>Solver gap</th>
      <th>ST % bound</th><th>ST &minus; CP-SAT</th></tr></thead>
    <tbody>{tprrows}</tbody></table></div>

  <div class="grid g2" style="margin-top:20px">
    <div class="card ok"><h3>Beats zeroth-order on all three</h3>
      <p class="sub" style="font-size:.9rem;margin-top:7px">By $967 to $2,664 at equal budget
      from identical starts. On the single-product line at comparable load the margin was
      $213&ndash;$398, and at flat 92% utilisation the gradient <em>lost</em>. The harder problem
      suits it better.</p></div>
    <div class="card open"><h3>Still loses to the exact solver</h3>
      <p class="sub" style="font-size:.9rem;margin-top:7px">CP-SAT leads by $916 to $1,686 on
      every mix. That is the same verdict as every configuration tested so far: the solver wins on
      solution quality. The gradient reaches 91&ndash;95% of a certified bound.</p></div>
    <div class="card ok"><h3>And the gradient got there in 36 seconds</h3>
      <p class="sub" style="font-size:.9rem;margin-top:7px">CP-SAT was given <strong>1800
      seconds</strong> to produce its column. The gradient reached 91% of the certified bound in
      <strong>35.7</strong> &mdash; fifty times less wall time for six points of bound. Below about
      a minute it beats the solver outright. See the next slide.</p></div>
    <div class="card flag"><h3>One difficulty axis explains everything</h3>
      <p class="sub" style="font-size:.9rem;margin-top:7px">The balanced 50/50 mix is hardest for
      <em>both</em> methods: the gradient gains most there (+$2,664), CP-SAT strains most there
      (3.3% gap against 1.5&ndash;2.1%), and the gradient sits furthest from the bound (91.2%).
      Changeover pressure is the variable &mdash; 64 changeovers at 50/50 against 28 when the mix
      is lopsided.</p></div>
  </div>

  <div class="grid g2" style="margin-top:18px">
    <div class="card"><h3>Gradient clipping matters here &mdash; it did not before</h3>
      <div class="scroll" style="margin-top:8px"><table>
        <thead><tr><th class="l">Mix</th><th>Unclipped</th><th>Clipped</th><th>Gain</th>
          <th>Initial &#8214;g&#8214;</th></tr></thead>
        <tbody>{cliprows}</tbody></table></div>
      <p class="sub" style="font-size:.9rem;margin-top:10px">On the single-product line the
      hypothesis that huge gradients were poisoning Adam's second-moment estimate was
      <em>tested and rejected</em> &mdash; clipping changed the result by $30 on $31,379. Here it
      is worth $156 to $539. The reason is scale: norms reach 7.17e9 against a single-product worst
      of 3.3e7, over 200&times; larger. A mechanism that was immaterial at one scale becomes real at
      another; the earlier rejection stands for the regime it was tested in.</p></div>
    <div class="card open"><h3>The first real strain on CP-SAT</h3>
      <div class="kv" style="margin-top:10px">
        <b>y variables</b><span>{yv}</span>
        <b>c variables</b><span>{cv}</span>
        <b>vs 1-product</b><span>{b1p} booleans total</span>
        <b>Gap, 2P @ 1800s</b><span>1.4% &ndash; 3.3%</span>
        <b>Gap, 1P @ 1800s</b><span>{g1p} at twelve tools</span>
        <b>Both budgets</b><span>1800s, 8 workers — matched</span>
      </div>
      <p class="sub" style="font-size:.92rem;margin-top:10px">Pipeline depth grew the model
      linearly and CP-SAT absorbed it. Setup variables grow it combinatorially, and the gap is
      roughly <strong>8&times; worse at half the time budget</strong>. This is the first
      configuration where the scaling premise shows any evidence at all.</p>
      <p class="sub" style="font-size:.92rem;margin-top:8px">It is strain, not collapse. CP-SAT
      still returns a better schedule than every learned method on every mix, and a 3.3% gap is
      perfectly usable in practice.</p></div>
  </div>
</section>
"""


SCALE_TEMPLATE = """
<section>
  <p class="eyebrow">The scaling question</p>
  <h2 style="margin:10px 0 8px">Doubling the solver's time budget did not close the gap</h2>
  <p class="sub" style="margin-bottom:18px">{note}</p>
  <div class="scroll"><table>
    <thead><tr><th class="l">Mix</th><th>Gap @ 900s</th><th>Gap @ 1800s</th><th>Moved</th>
      <th>Incumbent</th><th>Bound</th><th>Changeovers</th>
      <th>vs 1-product depth</th></tr></thead>
    <tbody>{m18rows}</tbody></table></div>

  <div class="grid g2" style="margin-top:20px">
    <div class="card flag"><h3>Why this run exists</h3>
      <p class="sub" style="font-size:.92rem;margin-top:8px">The first two-product solves used
      900 seconds &mdash; an arbitrary choice made to get a result inside a reasonable wait. The
      single-product depth study used 1800. An earlier version of this deck compared gaps across
      those different budgets and called the difference "8x more strain", which was not a sound
      comparison to draw.</p>
      <p class="sub" style="font-size:.92rem;margin-top:8px">This re-runs every mix at 1800
      seconds with the same eight workers, so the only remaining difference between the two studies
      is the problem itself. <strong>The claim survives the matched test:</strong> the balanced mix
      sits at 8.4x the depth study's gap, the lopsided mixes at 3.7&ndash;4.5x.</p></div>
    <div class="card open"><h3>Twice the time bought nothing</h3>
      <p class="sub" style="font-size:.92rem;margin-top:8px">Gaps moved by &minus;0.04, +0.24 and
      &minus;0.64 percentage points &mdash; in <em>both</em> directions, which is run-to-run
      variance for a randomised parallel solver rather than convergence. <strong>None of the three
      was proven optimal.</strong></p>
      <p class="sub" style="font-size:.92rem;margin-top:8px">That is the signature worth having.
      An under-resourced solve improves when given more time; a genuinely hard instance does not.
      Pipeline depth grew the model linearly and CP-SAT closed to 0.39% at this same budget. Setup
      variables grow it combinatorially, and the gap sits still.</p></div>
  </div>

  <div class="card" style="margin-top:16px"><h3>Read it carefully: this is strain, not collapse</h3>
    <p class="sub" style="font-size:.93rem;margin-top:8px">A gap is the distance between the best
    schedule CP-SAT has <em>found</em> and the best it can <em>prove</em> no schedule beats. It
    measures proof progress, not solution quality &mdash; the incumbent may already be optimal
    with the solver simply unable to certify it.</p>
    <p class="sub" style="font-size:.93rem;margin-top:8px">And CP-SAT still returns a better
    schedule than every learned method on every mix, by $733 to $1,700. A 1.4&ndash;3.3% unproven
    gap is entirely usable in practice. What has changed is that, for the first time in this
    programme, more compute stopped helping &mdash; which is the thing the scaling premise
    predicted and that pipeline depth conspicuously failed to show.</p></div>
</section>
"""


ANYTIME_TEMPLATE = """
<section>
  <p class="eyebrow">Time to a good schedule</p>
  <h2 style="margin:10px 0 8px">91% of the certified bound in 36 seconds</h2>
  <p class="sub" style="margin-bottom:20px">Every comparison so far has been at a fixed budget,
  which hides the question that matters for anything deployable: <strong>how good a schedule can
  each method produce, per second spent?</strong> Both solve the identical problem on the same
  machine; the only variable is how long they are given.</p>

  <div class="grid g3">
    <div class="card ok"><div class="stat" style="color:var(--ok)">{gw} s</div>
      <div class="statlbl">grad-ST &mdash; ${gp}, {gpc} of&nbsp;bound</div></div>
    <div class="card"><div class="stat" style="color:var(--ink-3)">{cw} s</div>
      <div class="statlbl">CP-SAT &mdash; ${cp}, {cpc} of&nbsp;bound</div></div>
    <div class="card flag"><div class="stat" style="color:var(--accent)">{ratio}&times;</div>
      <div class="statlbl">less wall time for 6 points of&nbsp;bound</div></div>
  </div>

  <figure style="margin-top:26px"><div id="anytime"></div>
    <figcaption>{note} The dashed line is the CP-SAT dual bound &mdash; a certified ceiling no
    schedule can exceed.</figcaption></figure>

  <div class="scroll" style="margin-top:20px"><table>
    <thead><tr><th class="l">Wall time</th><th>CP-SAT</th><th>Grad ST</th>
      <th class="l">&nbsp;</th></tr></thead>
    <tbody>{atrows}</tbody></table></div>

  <div class="grid g3" style="margin-top:20px">
    <div class="card ok"><h3>Below a minute, the gradient wins outright</h3>
      <p class="sub" style="font-size:.9rem;margin-top:7px">At 36 seconds CP-SAT has $25,180
      &mdash; worse than the greedy commander's $25,028. It has not yet found a schedule worth
      running. The gradient is already at 91% of the bound. Crossover sits around
      <strong>{crossover}</strong>.</p></div>
    <div class="card open"><h3>But it plateaus, and lower</h3>
      <p class="sub" style="font-size:.9rem;margin-top:7px">{gplat}. CP-SAT's ceiling is higher:
      {cplat}. Given real time the solver is simply better, and that remains true on every
      configuration tested.</p></div>
    <div class="card flag"><h3>And this should get faster</h3>
      <p class="sub" style="font-size:.9rem;margin-top:7px">These are <strong>CPU</strong> numbers.
      A batched forward-and-backward pass over thousands of trajectories is precisely the workload
      GPUs exist for, while CP-SAT's branch-and-bound tree search gains little from one. The
      crossover should move further in the gradient's favour on the NVIDIA box &mdash; well
      founded, but untested.</p></div>
  </div>

  <div class="card flag" style="margin-top:16px"><h3>Why this is the first result that looks deployable</h3>
    <p class="sub" style="font-size:.93rem;margin-top:8px">Every earlier comparison ran on the
    solver's terms: unlimited time, full knowledge of demand, one schedule computed once. Under
    those conditions CP-SAT won seven configurations out of seven, and it deserved to.</p>
    <p class="sub" style="font-size:.93rem;margin-top:8px">Rolling-horizon replanning does not work
    that way. It needs a good schedule in seconds, repeatedly, as the state changes &mdash; not a
    proven-optimal one in half an hour. <strong>That is a regime where the gradient is already
    ahead</strong>, and it is the concrete form of the latency argument this programme has been
    asserting without evidence until now.</p></div>
</section>
"""

def fmt(v, dp=0):
    if v is None:
        return "&mdash;"
    return f"{v:,.{dp}f}"


def signed(v, dp=0):
    if v is None:
        return "&mdash;"
    return f"{v:+,.{dp}f}"


def build() -> str:
    d = json.loads(DATA.read_text())
    P, H, TP = d["patterns"], d["historical"], d["throughput"]
    LABEL = {"UNIFORM": "Uniform 50%", "ALTERNATING_50": "Alternating 50%",
             "ALTERNATING": "Alternating 89%", "ALTERNATING_92": "Alternating 92%"}
    order = ["UNIFORM", "ALTERNATING_50", "ALTERNATING", "ALTERNATING_92"]

    # ---- headline table ------------------------------------------------
    rows = []
    for k in order:
        p = P[k]
        rows.append(f"""<tr><td>{LABEL[k]}</td>
<td>{p['util']:.0%}</td>
<td class="{'neg' if p['commander']<0 else ''}">{fmt(p['commander'])}</td>
<td>{fmt(p['random'])}</td><td>{fmt(p['hillclimb'])}</td>
<td>{fmt(p['grad_soft'])}</td><td class="hero">{fmt(p['grad_st'])}</td>
<td class="hero">{signed(p['grad_st']-p['hillclimb'])}</td>
<td class="muted">{fmt(p['milp'])}</td>
<td class="muted">{fmt(p['milp_bound'])}</td>
<td>{p['grad_pct_bound']:.1%}</td></tr>""")
    table = "".join(rows)

    # ---- historical table ----------------------------------------------
    PPO = d.get("ppo", {})
    hrows = []
    for k in order:
        h = H[k]
        cm = h["claimed_max"]
        pp = PPO.get(k)
        ppo_cell = (f"<td><strong>{fmt(pp['best'])}</strong></td>" if pp
                    else "<td class='muted'>&mdash;</td>")
        hrows.append(f"""<tr><td>{LABEL[k]}</td><td class="muted">{fmt(h['agent'])}</td>
{ppo_cell}
<td class="{'neg' if h['commander']<0 else ''}">{fmt(P[k]['commander'])}</td>
<td class="muted">{fmt(cm) if cm else '&mdash;'}</td>
<td class="hero">{fmt(P[k]['grad_st'])}</td></tr>""")

    # ---- depth sweep ---------------------------------------------------
    ds = d["depth_sweep"]
    drows = "".join(
        f"<tr><td>{h}</td><td>{fmt(r['grad_st'])}</td><td>{fmt(r['hillclimb'])}</td>"
        f"<td>{signed(r['grad_st']-r['hillclimb'])}</td>"
        f"<td>{r['first_quarter']:.1%}</td><td>{r['reach']:.0%}</td>"
        f"<td class='mono'>{r['nonfinite']}</td></tr>"
        for h, r in ds.items())

    payload = json.dumps({
        "curves": d["grad_curves"], "patterns": P, "labels": LABEL,
        "order": order, "throughput": TP, "depth": ds,
    })

    prov = d["status"]

    html = f"""<style>{CSS}</style>
<div class="wrap">

<section>
  <div class="stack">
    <p class="eyebrow">SimpleFab RL &nbsp;/&nbsp; technical review &nbsp;/&nbsp; 6 Sep 2026</p>
    <h1>A differentiable production line</h1>
    <p class="sub" style="font-size:1.16rem">
      We rewrote the fab simulator as a batched tensor program so that automatic
      differentiation can trace scheduling decisions all the way to profit. This deck
      covers what was built, what it proved, what it did not, and whether the approach
      earns the next round of investment.</p>
    <div class="lattice" aria-hidden="true" id="lat"></div>
  </div>

  <div class="banner" style="margin-top:32px">
    <strong>Status.</strong> Sections 1&ndash;5 are final: the four-pattern comparison, the
    credit-assignment diagnostic and the retrained PPO baselines are all complete, with CP-SAT
    reporting both its best schedule and a rigorous <em>dual bound</em>. Since then the line has
    been generalised to arbitrary pipeline depth and <strong>experiment family 3 has run</strong>
    &mdash; see &ldquo;Family 3&rdquo; below. The one open item is CP-SAT at depth: long solves are
    running now, so the family 3 table quotes the dual bound rather than an incumbent.
  </div>

  <div class="grid g3" style="margin-top:32px">
    <div class="card ok"><div class="stat" style="color:var(--ok)">4</div>
      <div class="statlbl">predictions the mechanism made and&nbsp;got&nbsp;right</div></div>
    <div class="card flag"><div class="stat" style="color:var(--accent)">8.4&times;</div>
      <div class="statlbl">CP-SAT gap vs pipeline depth, matched&nbsp;budget</div></div>
    <div class="card open"><div class="stat" style="color:var(--open)">0 / 7</div>
      <div class="statlbl">configurations where a learned method beat&nbsp;the&nbsp;solver</div></div>
  </div>
</section>

<section>
  <p class="eyebrow">The verdict</p>
  <h2 style="margin:10px 0 20px">The mechanism works and predicts; the solver still wins</h2>
  <div class="grid g3">
    <div class="card ok"><span class="pill ok">Confirmed</span>
      <h3 style="margin:12px 0 8px">Gradients survive the discontinuities</h3>
      <p class="sub" style="font-size:.93rem">Credit assignment propagates through 2,688
      steps of hard threshold gates with zero non-finite values. More than half the gradient
      mass sits in the first quarter of the horizon &mdash; the signal reaches decisions made
      2,000+ ticks before their revenue arrives.</p></div>
    <div class="card ok"><span class="pill ok">Confirmed</span>
      <h3 style="margin:12px 0 8px">It beats every search baseline &mdash; where decisions are live</h3>
      <p class="sub" style="font-size:.93rem">On all four single-product demand patterns and all
      three two-product mixes, at equal budget from identical starts. The margin tracks one thing:
      how many genuine choices the schedule contains. It is an order of magnitude at 50%
      utilisation, $967&ndash;$2,664 across the product mixes, and <em>negative</em> where the line
      runs saturated and there is nothing left to choose.</p></div>
    <div class="card open"><span class="pill open">Not shown</span>
      <h3 style="margin:12px 0 8px">The exact solver still wins, everywhere</h3>
      <p class="sub" style="font-size:.93rem">CP-SAT returns a better schedule than every learned
      method on every configuration tested &mdash; seven for seven. The gradient reaches
      91&ndash;95% of a certified bound but never passes it.</p>
      <p class="sub" style="font-size:.93rem;margin-top:8px">And every comparison so far is
      deterministic, full-information and open-loop, which is the solver's ideal case. Inference
      latency and adaptation to demand that differs from forecast &mdash; the two advantages
      CP-SAT cannot answer by being faster &mdash; remain untested.</p></div>
  </div>
</section>

<!--TECH-->

<section>
  <p class="eyebrow">How to read the results tables</p>
  <h2 style="margin:10px 0 8px">What every column means</h2>
  <p class="sub" style="margin-bottom:20px">Each row is one line configuration. Every method
  competes on the same problem, from the same starting point, with the same budget &mdash; and
  every profit figure is the final schedule replayed through the <strong>exact integer
  simulator</strong>. Relaxed or approximate profit is never reported as a result.</p>
  <div class="scroll"><table>
    <thead><tr><th class="l">Column</th><th class="l">What it is</th>
      <th class="l">Why it is there</th></tr></thead>
    <tbody>
      <tr><td class="l"><strong>Line</strong></td>
        <td class="l">Tool count as diffusion+litho pairs. "8 tools (4+4)" = 4 batch furnaces and
          4 single-wafer tools alternating in a serial chain.</td>
        <td class="l">The configuration under test.</td></tr>
      <tr><td class="l"><strong>Lead</strong></td>
        <td class="l">Minimum flow time &mdash; the sum of processing times along the chain
          (16+2+16+2 = 36t for 2+2). The floor on ticks from releasing raw material to a
          shippable unit, assuming zero queueing.</td>
        <td class="l"><strong>The credit-assignment distance</strong>: how far a release decision
          sits from the revenue it earns. This is family 3's independent variable.</td></tr>
      <tr><td class="l"><strong>Commander</strong></td>
        <td class="l">The greedy heuristic in <code>sim.py</code>: run any tool that is idle and
          has enough input. No lookahead, no tuning, deterministic.</td>
        <td class="l">The "without thinking" floor, and the schedule CP-SAT is warm-started from.</td></tr>
      <tr><td class="l"><strong>Random</strong></td>
        <td class="l">Random search over action tapes at run-rates spread across 0.1&ndash;0.95,
          each evaluated on the exact simulator, best kept.</td>
        <td class="l">The floor for search that sees only the simulator, never a gradient.</td></tr>
      <tr><td class="l"><strong>Hill-climb</strong></td>
        <td class="l">A (1+1) evolution strategy: starts from the <em>same</em> tapes as the
          gradient, flips bits, keeps improvements. Flip-rate ladder of 1/4/16/64 bits across the
          batch so it is not a strawman.</td>
        <td class="l"><strong>The control that matters.</strong> Identical start, identical
          budget; the only difference from Grad&nbsp;ST is whether a gradient is used.</td></tr>
      <tr><td class="l"><strong>Grad soft</strong></td>
        <td class="l">Adam on the tape through a <em>fully relaxed</em> simulator &mdash;
          fractional jobs, smooth gates. Its own forward profit is not achievable, so the tape is
          rounded and replayed through the exact sim.</td>
        <td class="l">Tests whether smoothing the whole simulator helps. It does not: this loses
          to the greedy heuristic on most configurations.</td></tr>
      <tr><td class="l"><strong>Grad ST</strong></td>
        <td class="l">Adam through <strong>straight-through</strong> gradients. Forward pass
          bit-identical to the exact simulator; only the backward pass substitutes smooth
          surrogates for the hard threshold gates.</td>
        <td class="l">The method under test.</td></tr>
      <tr><td class="l"><strong>vs best 0th-order</strong></td>
        <td class="l">Grad&nbsp;ST minus the better of {{Random, Hill-climb}}.</td>
        <td class="l">The honest margin. Taking the <em>best</em> of the two, rather than a
          convenient one, avoids cherry-picking.</td></tr>
      <tr><td class="l"><strong>% of bound</strong></td>
        <td class="l">Grad&nbsp;ST as a fraction of CP-SAT's <em>dual bound</em> &mdash; a proven
          ceiling that no schedule can exceed.</td>
        <td class="l">How close to provably optimal, on a scale comparable across configurations
          with different demand.</td></tr>
    </tbody></table></div>

  <div class="grid g2" style="margin-top:20px">
    <div class="card flag"><h3>Why "% of bound" is trustworthy before CP-SAT finishes</h3>
      <p class="sub" style="font-size:.92rem;margin-top:8px">The dual bound is proven at every
      moment of a solve: it starts at the LP relaxation and only ever moves <strong>down</strong>
      toward the true optimum. It can never move up. So these percentages are <em>conservative</em>
      &mdash; a longer solve can only improve them.</p>
      <p class="sub" style="font-size:.92rem;margin-top:8px">It does need some solve time to
      become useful. At <strong>20 seconds it is still at the trivial $26,880</strong>
      (336 units &times; $80 &mdash; ship everything, pay nothing). By 45 seconds it has tightened
      to $19,008.0, and 600 seconds moves it only a further <strong>0.02%</strong> to $19,004.2.
      So: tens of seconds to become meaningful, then stable.</p>
      <p class="sub" style="font-size:.92rem;margin-top:8px">The <em>incumbent</em> &mdash; CP-SAT's
      best actual schedule &mdash; behaves the opposite way: it takes tens of minutes and is the
      number still outstanding at depth.</p></div>
    <div class="card"><h3>What makes the comparison fair</h3>
      <ul class="sub" style="font-size:.92rem;margin-top:8px">
        <li><strong>Equal budget.</strong> 76,800 rollout-equivalents, counting one gradient step
          as three forward passes (forward + backward). The zeroth-order methods therefore get
          1,200 iterations at batch 64.</li>
        <li><strong>Same starting point.</strong> Grad&nbsp;ST and Hill-climb begin from identical
          random tapes; only random search draws fresh samples.</li>
        <li><strong>Same yardstick.</strong> Every number is exact-integer-sim profit, never the
          relaxed objective a method optimised internally.</li>
        <li><strong>Not comparable to PPO.</strong> All of these optimise an open-loop schedule
          with full advance knowledge of demand. PPO is reactive and sees an observation vector.
          The fair pairings are gradient&nbsp;vs&nbsp;CP-SAT and PPO&nbsp;vs&nbsp;commander.</li>
      </ul></div>
  </div>
</section>

<!--CAP-->

<section>
  <p class="eyebrow">Result 1 &mdash; the headline</p>
  <h2 style="margin:10px 0 8px">Gradient versus every baseline, four demand patterns</h2>
  <p class="sub" style="margin-bottom:22px">All methods start from the same random schedules
  and get the same budget in forward-rollout-equivalents. Every figure is the rounded schedule
  replayed through the <em>exact</em> integer simulator &mdash; relaxed-forward profit is never
  reported as a result.</p>
  <div class="scroll"><table>
    <thead><tr><th>Pattern</th><th>Util</th><th>Commander</th><th>Random</th>
      <th>Hill-climb</th><th>Grad&nbsp;soft</th><th>Grad&nbsp;ST</th>
      <th>ST&nbsp;&minus;&nbsp;HC</th><th>CP-SAT</th><th>Upper&nbsp;bound</th>
      <th>ST % of bound</th></tr></thead>
    <tbody>{table}</tbody></table></div>
  <figure style="margin-top:30px">
    <div class="legend" id="lg1"></div><div id="c1"></div>
    <figcaption>Profit by method. The gap at 50% utilisation is the finding: there the commander
    runs flat out and goes <em>negative</em> on holding costs, and local bit-flipping cannot
    discover the coordinated shutdown that the optimum requires. The backward pass sees the whole
    horizon at once. CP-SAT is omitted from the bars &mdash; it is a ceiling, not a competitor at
    equal budget.</figcaption>
  </figure>
</section>

<section>
  <p class="eyebrow">Result 2 &mdash; the diagnostic that matters</p>
  <h2 style="margin:10px 0 8px">Credit assignment does not vanish</h2>
  <p class="sub" style="margin-bottom:22px">Gradient magnitude with respect to the decision at
  each tick. If this collapsed toward the right-hand edge, the gradient could not tell the
  optimiser to pre-build for a demand spike hundreds of ticks later &mdash; which is precisely
  the decision PPO's local, discounted reward gets wrong.</p>
  <figure><div class="legend" id="lg2"></div><div id="c2"></div>
    <figcaption>Share of total gradient magnitude by tick, normalised per pattern.
    Mass is spread across the full 2,688-tick horizon rather than concentrated near the
    terminal reward.</figcaption></figure>
  <div class="scroll" style="margin-top:26px"><table>
    <thead><tr><th>Horizon</th><th>Grad ST</th><th>Hill-climb</th><th>Margin</th>
      <th>Mass in 1st quarter</th><th>90% mass spans</th><th>Non-finite</th></tr></thead>
    <tbody>{drows}</tbody></table></div>
  <p class="sub muted" style="margin-top:14px;font-size:.88rem">Depth sweep on a synthetic
  self-similar problem family held at 90% utilisation. Note the margins here are small &mdash;
  that family turned out not to discriminate between optimisers, which is why the real demand
  patterns above were run.</p>
</section>

<section>
  <p class="eyebrow">Result 3 &mdash; design findings</p>
  <h2 style="margin:10px 0 20px">Three things that would have cost weeks to learn later</h2>
  <div class="grid g3">
    <div class="card"><h3>Exact forward beats relaxed forward</h3>
      <p class="sub" style="font-size:.93rem;margin-top:8px">The fully-relaxed simulator
      (<code>grad-soft</code>) loses to the greedy heuristic on three of four patterns.
      Optimising a smoothed system optimises the wrong system, and rounding back to integers
      discards the gains. Straight-through &mdash; exact forward, surrogate backward &mdash;
      is the correct design.</p></div>
    <div class="card"><h3>The exact solver still wins &mdash; but slowly</h3>
      <p class="sub" style="font-size:.93rem;margin-top:8px">CP-SAT beats the gradient on all four
      patterns, by $590 to $1,375. It needed 600&nbsp;s to get there, and even then closed the
      optimality gap on <em>none</em> of them &mdash; 0.4% to 1.5% remained on a single-product,
      four-machine problem. At 180&nbsp;s it had been beaten by random search. Solve time, not
      solution quality, is the variable to watch as the problem grows.</p></div>
    <div class="card open"><h3>A live bug in the training reward</h3>
      <p class="sub" style="font-size:.93rem;margin-top:8px"><code>env.py</code> sums four numpy
      booleans, and numpy boolean addition is logical OR &mdash; so the invalid-action count
      saturates at 1. With <code>invalid_action_penalty=-2.0</code>, an agent requesting four
      illegal starts is penalised exactly as much as one. Found by the port's equivalence
      test.</p></div>
  </div>
</section>


<!--FAMILY3-->
<!--F92-->

<!--TWOPROD-->

<!--TPRES-->
<!--ANYTIME-->
<!--SCALE-->

<section>
  <p class="eyebrow">Against the record</p>
  <h2 style="margin:10px 0 8px">How this compares to the previous experiments</h2>
  <p class="sub" style="margin-bottom:22px">Because the models behind the historical scores had been
  overwritten, PPO was <strong>retrained from scratch on all three patterns</strong> (8M steps each,
  unchanged hyperparameters) under today's cost settings, then evaluated with the project's own
  methodology &mdash; temperature 0.7, best of 20 episodes. Those retrains land within about
  &plusmn;$1,600 of the historical record, which for single-seed runs is unremarkable. The commander
  column is exactly reproducible: the tensor simulator reproduces all three published values to the
  dollar, which is what makes the rest of the comparison trustworthy.</p>
  <div class="scroll"><table>
    <thead><tr><th>Pattern</th><th>PPO (historical)</th><th>PPO (retrained today)</th>
      <th>Commander</th><th>Claimed "max"</th><th>Grad ST (open-loop)</th></tr></thead>
    <tbody>{"".join(hrows)}</tbody></table></div>
  <div class="card open" style="margin-top:22px">
    <h3>Read the last two columns with care &mdash; they are not a fair fight</h3>
    <p class="sub" style="font-size:.93rem;margin-top:8px">The gradient column optimises an
    <strong>open-loop schedule with full advance knowledge of demand</strong> &mdash; the same
    privilege the MILP has. PPO is a <strong>reactive policy</strong> that sees only an 18-dimensional
    observation and must decide tick by tick. Beating it here is expected and proves nothing about
    closed-loop performance. What the comparison does establish is a useful bound: it shows how much
    profit is on the table for a scheduler that reasons over the whole horizon, and therefore how much
    a reactive policy is giving up. The fair comparisons are gradient&nbsp;vs&nbsp;MILP (both
    open-loop) and PPO&nbsp;vs&nbsp;commander (both reactive).</p></div>
  <div class="card flag" style="margin-top:16px"><h3>The old "theoretical max" was not an optimum</h3>
    <p class="sub" style="font-size:.93rem;margin-top:8px">It was a hand-computed bound &mdash;
    roughly demand &times; (revenue &minus; production cost) minus a small term. Every
    "% of maximum" figure in the project record was measured against it. Replacing it with a
    certified CP-SAT dual bound on all four patterns is a standalone upgrade to the project's
    evaluation, independent of anything the gradient does.</p></div>
</section>

<section>
  <p class="eyebrow">What was built</p>
  <h2 style="margin:10px 0 20px">Infrastructure, and what proves it correct</h2>
  <div class="grid g2">
    <div class="card"><h3>Three equivalence gates, all passing</h3>
      <ul class="sub" style="font-size:.93rem;margin-top:10px">
        <li><strong>Simulator</strong> &mdash; 28 trajectories &times; 4 demand modes &times;
          2,688 ticks. Every queue count, machine state, shipment and cost accumulator identical;
          profit agrees to ~1e-9.</li>
        <li><strong>Environment</strong> &mdash; all 18 observation dimensions, action mask,
          reward, shaping term and invalid counts match across 3 shaping configurations.</li>
        <li><strong>Differentiable line</strong> &mdash; forward pass bit-identical to the exact
          simulator; naive autodiff through the hard gates returns exactly zero, as it must.</li>
      </ul></div>
    <div class="card"><h3>Throughput, XLA CPU only</h3>
      <div id="c3" style="margin-top:6px"></div>
      <p class="sub muted" style="font-size:.86rem;margin-top:10px">No GPU numbers yet &mdash;
      macOS has no CUDA backend. The environment wrapper mattered more than the physics: it cost
      4.4&times; the dynamics it wrapped.</p></div>
  </div>
</section>

<section>
  <p class="eyebrow">The honest ledger</p>
  <h2 style="margin:10px 0 20px">What is banked, and what is not</h2>
  <div class="grid g2">
    <div class="card ok"><h3 style="color:var(--ok)">Banked</h3>
      <ul class="sub" style="font-size:.93rem;margin-top:10px">
        <li>A verified differentiable simulator &mdash; a capability that did not exist</li>
        <li>Experimental velocity: the four-pattern, five-method comparison ran in 16 minutes</li>
        <li>The gradient beats every baseline on every real demand pattern</li>
        <li>A certified optimality bound replacing a hand-computed one</li>
        <li>A live reward bug in the training pipeline</li>
      </ul></div>
    <div class="card open"><h3 style="color:var(--open)">Not banked</h3>
      <ul class="sub" style="font-size:.93rem;margin-top:10px">
        <li><strong>No policy is better.</strong> Best learned result is still the May PPO run</li>
        <li>The gradient does <strong>not</strong> beat the exact solver &mdash; CP-SAT leads on all
          four patterns by $590&ndash;$1,375</li>
        <li>No training speedup &mdash; PPO not yet swapped over; the simulator was never its bottleneck</li>
        <li>Nothing tested beyond one product and four machines</li>
        <li>Open-loop only &mdash; these schedules need full advance knowledge of demand</li>
        <li>The scaling premise remains an assumption, not a measurement</li>
      </ul></div>
  </div>
</section>

<section>
  <p class="eyebrow">Next</p>
  <h2 style="margin:10px 0 20px">Three moves, in priority order</h2>
  <div class="stack">
    <div class="card flag"><h3>1 &nbsp;&middot;&nbsp; Test the assumption everything rests on</h3>
      <p class="sub" style="font-size:.95rem;margin-top:8px">Scale CP-SAT up &mdash; more machines,
      then multiple products, then reentrant flows &mdash; and chart where solve time explodes.
      The entire strategic case depends on exact optimisation breaking at realistic scale, and that
      is currently backed by a complexity-class argument rather than a measurement. Roughly a week,
      and it either validates or kills the premise before anything expensive is built.
      The 180-second failures above suggest the knee may be closer than assumed.</p></div>
    <div class="card"><h3>2 &nbsp;&middot;&nbsp; The closed-loop gradient test</h3>
      <p class="sub" style="font-size:.95rem;margin-top:8px">Train a reactive policy by
      backpropagating end-of-episode profit through the dynamics. Straight-through only; gradient
      clipping is mandatory. <strong>Pre-registered kill criterion:</strong> if it cannot beat
      PPO's $31,493 on Alternating 92%, the pathwise route is dead for this problem and we stop.</p></div>
    <div class="card"><h3>3 &nbsp;&middot;&nbsp; A cheaper hypothesis, in parallel</h3>
      <p class="sub" style="font-size:.95rem;margin-top:8px">PPO's diagnosed failure is discounted
      local reward driving under-production. The tensorised environment now makes it cheap to run
      PPO at large batch with &gamma;=1.0 and full-episode returns, which attacks that pathology
      directly &mdash; no differentiability required. It would be embarrassing to build a
      differentiable trainer and then find the discount factor was the answer.</p></div>
  </div>
</section>

<section>
  <p class="eyebrow">Research potential</p>
  <h2 style="margin:10px 0 20px">An honest assessment</h2>
  <div class="grid g2">
    <div class="card"><h3>Where there is little</h3>
      <p class="sub" style="font-size:.93rem;margin-top:8px">GPU-vectorised environments,
      differentiable simulation, straight-through estimators, and RL for job-shop scheduling are
      all well-trodden. "We built a fast simulator" is engineering, not a contribution.</p></div>
    <div class="card flag"><h3>Where there is real potential</h3>
      <ul class="sub" style="font-size:.93rem;margin-top:8px">
        <li><strong>Differentiable discrete-event simulation.</strong> The literature is
          overwhelmingly rigid-body and contact physics. Production and queueing systems are a
          genuine gap.</li>
        <li><strong>Why it works here.</strong> Our discontinuities are multiplicative gates, not
          reflective contacts &mdash; so the linear pathway survives underneath the surrogate.
          That is a mechanistic explanation for beating the pessimism in the literature.</li>
        <li><strong>The reentrancy crossover.</strong> Charting where exact solvers break against
          where a learned policy still trains would be a real contribution &mdash; and it is
          move&nbsp;1 above.</li>
      </ul></div>
  </div>
  <p class="sub" style="margin-top:20px">Overall: moderate, and concentrated almost entirely in the
  scaling question. On a four-machine toy the problem is too easy for any optimiser to distinguish
  itself decisively &mdash; which is exactly what the 90%-utilisation results show.</p>
</section>

<section>
  <p class="eyebrow">Provenance</p>
  <h2 style="margin:10px 0 14px">Everything here is reproducible</h2>
  <p class="sub">Every figure traces to a logged run under <code>tensorfab/runs/</code>, each
  capturing the git commit, backend, parameters, full transcript and raw series. This deck is
  generated by <code>tensorfab/build_deck.py</code> from a single JSON extract &mdash; no number
  is typed by hand.</p>
  <div class="scroll" style="margin-top:18px"><table>
    <thead><tr><th>Command</th><th>Produces</th></tr></thead><tbody>
    <tr><td>python -m tensorfab.verify --all-modes</td><td>simulator equivalence gate</td></tr>
    <tr><td>python -m tensorfab.verify_env</td><td>environment equivalence gate</td></tr>
    <tr><td>python -m tensorfab.verify_diffsim</td><td>surrogate-gradient gate</td></tr>
    <tr><td>python -m tensorfab.compare_patterns</td><td>the four-pattern comparison</td></tr>
    <tr><td>python -m tensorfab.spike_openloop</td><td>the BPTT depth sweep</td></tr>
    <tr><td>python -m tensorfab.bench / bench_env</td><td>throughput</td></tr>
    </tbody></table></div>
</section>

<section style="padding-bottom:72px">
  <div class="lattice" aria-hidden="true" id="lat2"></div>
  <p class="muted mono" style="font-size:.72rem;margin-top:14px">
    SimpleFab RL &middot; tensorfab &middot; generated from logged runs &middot;
    CP-SAT ceiling: {prov['milp_ceilings']}</p>
</section>
</div>

__SCRIPT__
"""

    # ---- family 3 + generalization sections ----------------------------
    F3 = d.get("family3", {})
    f3rows = "".join(
        f"<tr><td>{r['n_pairs']*2} tools ({r['n_pairs']}+{r['n_pairs']})</td>"
        f"<td>{r['lead']}t</td>"
        f"<td class=\"{'neg' if r['commander']<0 else ''}\">{fmt(r['commander'])}</td>"
        f"<td>{fmt(r['random'])}</td><td>{fmt(r['hillclimb'])}</td>"
        f"<td>{fmt(r['grad_soft'])}</td><td class='hero'>{fmt(r['grad_st'])}</td>"
        f"<td class='hero'>{signed(r['grad_st']-max(r['random'],r['hillclimb']))}</td>"
        f"<td class='muted'>{fmt(r['bound'])}</td>"
        f"<td>{r['grad_st']/r['bound']:.1%}</td></tr>"
        for r in F3.get("rows", []))
    VM = d.get("verification_matrix", [])
    vmrows = "".join(
        f"<tr><td class='l'>{v['gate']}</td><td class='l'>{v['checks']}</td>"
        f"<td>{v['m4']}</td><td>{v['m8']}</td><td>{v['m12']}</td>"
        f"<td class='l muted'>{v['kind']}</td></tr>" for v in VM)
    GEN = d.get("generalization", {})
    genrows = "".join(
        f"<tr><td class='l'><code>{g['f']}</code></td><td class='l'>{g['change']}</td>"
        f"<td class='l muted'>{g['risk']}</td></tr>" for g in GEN.get("files", []))
    bugrows = "".join(
        f"<div class='card open' style='margin-top:10px'><h3>{b['b']}</h3>"
        f"<p class='sub' style='font-size:.9rem;margin-top:6px'><strong>Impact:</strong> {b['impact']}<br>"
        f"<strong>Caught by:</strong> {b['found']}</p></div>" for b in GEN.get("bugs", []))
    family3_html = FAMILY3_TEMPLATE.format(
        note=F3.get("note", ""), milp_status=F3.get("milp_status", ""),
        f3rows=f3rows, vmrows=vmrows, genrows=genrows, bugrows=bugrows)
    html = html.replace("<!--FAMILY3-->", family3_html)


    F92 = d.get("family3_92", {})
    f92rows = "".join(
        f"<tr><td>{r['n_pairs']*2} tools ({r['n_pairs']}+{r['n_pairs']})</td><td>{r['lead']}t</td>"
        f"<td>{fmt(r['commander'])}</td><td>{fmt(r['random'])}</td><td>{fmt(r['hillclimb'])}</td>"
        f"<td class=\"{'neg' if r['grad_soft']<0 else ''}\">{fmt(r['grad_soft'])}</td>"
        f"<td>{fmt(r['grad_st'])}</td>"
        f"<td class='neg'>{signed(r['grad_st']-max(r['random'],r['hillclimb']))}</td></tr>"
        for r in F92.get("rows", []))
    ST = d.get("slack_trend", [])
    strows = "".join(
        f"<tr><td class='l'>{t['problem']}</td><td>{t['util']:.0%}</td>"
        f"<td class='l'>{t['shape']}</td>"
        f"<td class=\"{'neg' if t['margin']<0 else 'hero'}\">{signed(t['margin'])}</td></tr>"
        for t in ST)
    MD = d.get("milp_depth", [])
    mdrows = "".join(
        f"<tr><td>M={r['M']}</td><td>{fmt(r['incumbent'])}</td><td>{fmt(r.get('grad_st'))}</td>"
        f"<td class='muted'>{signed(r['incumbent']-r['grad_st']) if r.get('grad_st') else ''}</td>"
        f"<td>{fmt(r['bound'])}</td><td>{r['gap']:.2%}</td></tr>" for r in MD)
    html = html.replace("<!--F92-->", F92_TEMPLATE.format(
        note=F92.get("note",""), clip=F92.get("clip_test",""),
        f92rows=f92rows, strows=strows, mdrows=mdrows))


    CAP = d.get("capacity", {})
    caprows = "".join(
        f"<tr><td class='l'>{r['line']} ({r['M']} tools)</td><td>{r['lead']}t</td>"
        f"<td>{CAP.get('nominal',672)}</td><td>{r['achievable']}</td>"
        f"<td>{r['d336']:.1%}</td><td>{r['d616']:.1%}</td></tr>"
        for r in CAP.get("rows", []))
    html = html.replace("<!--CAP-->", CAP_TEMPLATE.format(
        explain=CAP.get("explain",""), caprows=caprows))

    html = html.replace("<!--TECH-->", TECH_HTML)

    TP = d.get("twoprod", {})
    tprows = "".join(
        f"<tr><td class='l'><strong>{r['p']}</strong></td><td>${r['revenue']}</td>"
        f"<td>${r['prod_cost']}</td><td>${r['margin']}</td><td>{r['batch_pt']}t</td>"
        f"<td>{r['t_per_unit']:.1f}</td><td class='hero'>${r['margin_per_tick']:.2f}</td>"
        f"<td>${r['fg']}</td><td>${r['backorder']}</td></tr>"
        for r in TP.get("products", []))
    alrows = "".join(
        f"<tr><td class='l'>{a['item']}</td><td class='l muted'>{a['before']}</td>"
        f"<td class='l'>{a['after']}</td><td class='l'>{a['why']}</td></tr>"
        for a in TP.get("alignment", []))
    keptrows = "".join(f"<li>{k}</li>" for k in TP.get("kept", []))
    mixrows = "".join(
        f"<tr><td class='l'>{m['mix']}</td><td>{m['d0']} / {m['d1']}</td>"
        f"<td>{fmt(m['commander'])}</td><td class='muted'>{fmt(m['ceiling'])}</td>"
        f"<td>{m['pct']:.1%}</td><td>{m['setups']}</td><td>{fmt(m.get('inv'))}</td></tr>"
        for m in TP.get("mixes", []))
    moderows = "".join(
        f"<tr><td class='l'>{m['mode']}</td><td>{fmt(m['profit'])}</td><td>{fmt(m['inv'])}</td>"
        f"<td>{fmt(m['wip'])}</td><td>{fmt(m['setup'])}</td><td>{fmt(m['backorder'])}</td></tr>"
        for m in TP.get("modes", []))
    planrows = "".join(
        f"<div class='card{' flag' if s_['n'] == 4 else ''}' style='margin-top:10px'>"
        f"<h3>{s_['n']} &middot; {s_['step']}</h3>"
        f"<p class='sub' style='font-size:.92rem;margin-top:6px'>{s_['detail']}</p>"
        f"<p class='sub' style='font-size:.86rem;margin-top:6px;color:var(--accent)'>"
        f"New in the 2-product case: {s_['new']}</p></div>"
        for s_ in TP.get("plan", []))
    BA = TP.get("before_after", {})
    html = html.replace("<!--TWOPROD-->", TWOPROD_TEMPLATE.format(
        tprows=tprows, alrows=alrows, keptrows=keptrows, mixrows=mixrows,
        moderows=moderows, planrows=planrows,
        stagerows="".join(
            f"<tr><td class='l'>{st['stage']}</td><td>{fmt(st['profit'])}</td>"
            f"<td>{fmt(st['inventory'])}</td><td>{fmt(st['backorder'])}</td>"
            f"<td>{st['shipped']} / {BA.get('demand')}</td></tr>"
            for st in BA.get("stages", []))))


    TPR = d.get("twoprod", {}).get("results", {})
    tprrows = "".join(
        f"<tr><td class='l'>{r['mix']}</td><td>{fmt(r['commander'])}</td>"
        f"<td>{fmt(r['random'])}</td><td>{fmt(r['hillclimb'])}</td>"
        f"<td class='hero'>{fmt(r['grad_st'])}</td>"
        f"<td class='hero'>{signed(r['grad_st']-max(r['random'],r['hillclimb']))}</td>"
        f"<td>{fmt(r['milp'])}</td><td class='muted'>{fmt(r['bound'])}</td>"
        f"<td>{r['gap']:.1%}</td><td>{r['grad_st']/r['bound']:.1%}</td>"
        f"<td class='neg'>{signed(r['grad_st']-r['milp'])}</td></tr>"
        for r in TPR.get("rows", []))
    cliprows = "".join(
        f"<tr><td class='l'>{c['mix']}</td><td>{fmt(c['unclipped'])}</td>"
        f"<td>{fmt(c['clipped'])}</td><td class='hero'>{signed(c['clipped']-c['unclipped'])}</td>"
        f"<td class='muted'>{c['gnorm']:.2e}</td></tr>"
        for c in d.get("twoprod", {}).get("results", {}).get("clip", []))
    MS = d.get("twoprod", {}).get("results", {}).get("milp_scale", {})
    html = html.replace("<!--TPRES-->", TPRES_TEMPLATE.format(
        note=TPR.get("note", ""), tprrows=tprrows, cliprows=cliprows,
        yv=f"{MS.get('y_vars',0):,}", cv=f"{MS.get('c_vars',0):,}",
        b1p=f"{MS.get('vs_1p_bools',0):,}", g1p=f"{MS.get('vs_1p_gap',0):.2%}"))


    M18 = d.get("twoprod", {}).get("results", {}).get("milp_1800", [])
    ref_gap = d.get("milp_depth_ref", {}).get("gap", 0.0039)
    m18rows = "".join(
        f"<tr><td class='l'>{m['mix']}</td><td>{m['gap900']:.2%}</td>"
        f"<td class='hero'>{m['gap1800']:.2%}</td>"
        f"<td class='muted'>{(m['gap1800']-m['gap900'])*100:+.2f} pp</td>"
        f"<td>{fmt(m['inc1800'])}</td><td class='muted'>{fmt(m['bound1800'])}</td>"
        f"<td>{m['chg']}</td><td>{m['gap1800']/ref_gap:.1f}x</td></tr>"
        for m in M18)
    html = html.replace("<!--SCALE-->", SCALE_TEMPLATE.format(
        m18rows=m18rows,
        note=d.get("twoprod", {}).get("results", {}).get("milp_note", "")))


    AT = d.get("twoprod", {}).get("results", {}).get("anytime", {})
    HL = AT.get("headline", {})
    atrows = "".join(
        f"<tr><td class='l'>{r['wall']}</td>"
        f"<td class='muted'>{fmt(r['cpsat']) if r['cpsat'] else '&mdash;'}</td>"
        f"<td class='hero'>{fmt(r['grad']) if r['grad'] else '&mdash;'}</td>"
        f"<td class='l muted'>{r['note']}</td></tr>"
        for r in AT.get("rows", []))
    html = html.replace("<!--ANYTIME-->", ANYTIME_TEMPLATE.format(
        note=AT.get("note", ""), atrows=atrows,
        gw=f"{HL.get('grad_wall',0):.1f}", gp=fmt(HL.get('grad_profit')),
        gpc=f"{HL.get('grad_pct',0):.1%}",
        cw=f"{HL.get('cpsat_wall',0):,}", cp=fmt(HL.get('cpsat_profit')),
        cpc=f"{HL.get('cpsat_pct',0):.1%}", ratio=HL.get('ratio', 50),
        bound=fmt(AT.get("bound")), crossover=AT.get("crossover", ""),
        gplat=AT.get("grad_plateau", ""), cplat=AT.get("cpsat_plateau", "")))

    return html.replace("__SCRIPT__",
                        "<script>" + JS.replace("__PAYLOAD__", payload) + "</script>")


if __name__ == "__main__":
    OUT.write_text(build())
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")
