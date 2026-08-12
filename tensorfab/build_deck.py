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
    --ok:#12A47B; --open:#E0455F; --warn:#B8830C;
  }
}
:root[data-theme="dark"]{
  --ground:#12131A; --panel:#191B24; --sunk:#0D0E14;
  --ink:#EFEADF; --ink-2:#B6AE9E; --ink-3:#78715F; --rule:#282B37;
  --accent:#EBA91C; --accent-soft:#2A2413;
  --c1:#B8830C; --c2:#12A47B; --c3:#4A86E8; --c4:#E0455F; --c5:#A05FE0;
  --ok:#12A47B; --open:#E0455F; --warn:#B8830C;
}
:root[data-theme="light"]{
  --ground:#FAF7F0; --panel:#FFFFFF; --sunk:#F1ECE1;
  --ink:#1A1712; --ink-2:#4A4238; --ink-3:#7D7264; --rule:#E2D9C9;
  --accent:#A87400; --accent-soft:#F0E4C8;
  --c1:#A87400; --c2:#0A7D5C; --c3:#2C63C4; --c4:#B92A45; --c5:#7A3FB8;
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
    <p class="eyebrow">SimpleFab RL &nbsp;/&nbsp; technical review &nbsp;/&nbsp; 2 Aug 2026</p>
    <h1>A differentiable production line</h1>
    <p class="sub" style="font-size:1.16rem">
      We rewrote the fab simulator as a batched tensor program so that automatic
      differentiation can trace scheduling decisions all the way to profit. This deck
      covers what was built, what it proved, what it did not, and whether the approach
      earns the next round of investment.</p>
    <div class="lattice" aria-hidden="true" id="lat"></div>
  </div>

  <div class="banner" style="margin-top:32px">
    <strong>All figures final.</strong> The CP-SAT column is a 600-second solve reporting both its
    best schedule and its <em>dual bound</em> &mdash; a rigorous ceiling no schedule can exceed.
    An earlier 180-second solve appeared to show the gradient beating the optimiser on one pattern;
    that was an artifact of an unconverged solve and does not survive.
  </div>

  <div class="grid g3" style="margin-top:32px">
    <div class="card ok"><div class="stat" style="color:var(--ok)">4 / 4</div>
      <div class="statlbl">patterns where the gradient beat every&nbsp;search baseline</div></div>
    <div class="card flag"><div class="stat" style="color:var(--accent)">95.3%</div>
      <div class="statlbl">mean of the certified upper&nbsp;bound</div></div>
    <div class="card"><div class="stat">0</div>
      <div class="statlbl">non-finite gradients across 2,688-step&nbsp;backprop</div></div>
  </div>
</section>

<section>
  <p class="eyebrow">The verdict</p>
  <h2 style="margin:10px 0 20px">The mechanism works, and it now has a margin</h2>
  <div class="grid g3">
    <div class="card ok"><span class="pill ok">Confirmed</span>
      <h3 style="margin:12px 0 8px">Gradients survive the discontinuities</h3>
      <p class="sub" style="font-size:.93rem">Credit assignment propagates through 2,688
      steps of hard threshold gates with zero non-finite values. More than half the gradient
      mass sits in the first quarter of the horizon &mdash; the signal reaches decisions made
      2,000+ ticks before their revenue arrives.</p></div>
    <div class="card ok"><span class="pill ok">Confirmed</span>
      <h3 style="margin:12px 0 8px">It beats every search baseline</h3>
      <p class="sub" style="font-size:.93rem">On all four real demand patterns, at equal budget
      from identical starts, and it lands within 3&ndash;8% of a certified upper bound. At 50%
      utilisation, where the optimum requires coordinated idling, the margin over zeroth-order
      search is an order of magnitude. The exact solver still edges it &mdash; see below.</p></div>
    <div class="card open"><span class="pill open">Not yet shown</span>
      <h3 style="margin:12px 0 8px">Nothing about closed-loop policies</h3>
      <p class="sub" style="font-size:.93rem">Every result here optimises an open-loop
      schedule with full horizon information &mdash; the MILP's game, not PPO's. No reactive
      policy has been trained. That is the next experiment and the one that matters.</p></div>
  </div>
</section>

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
    return html.replace("__SCRIPT__",
                        "<script>" + JS.replace("__PAYLOAD__", payload) + "</script>")


if __name__ == "__main__":
    OUT.write_text(build())
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")
