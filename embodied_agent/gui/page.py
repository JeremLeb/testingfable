"""The single-page dashboard (HTML + CSS + vanilla JS, no external deps)."""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Embodied Agent — live</title>
<style>
  :root{
    --bg:#0e1116; --panel:#171b22; --panel2:#1e242d; --ink:#e6edf3;
    --muted:#9aa7b4; --line:#2a323d; --accent:#4c8dff;
    --green:#39d98a; --blue:#4c8dff; --orange:#ffab4c; --red:#ff5c5c;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
  header{padding:16px 22px;border-bottom:1px solid var(--line);
    display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
  header h1{font-size:19px;margin:0}
  header .sub{color:var(--muted);font-size:13px}
  .badge{margin-left:auto;padding:4px 10px;border-radius:999px;
    background:var(--panel2);border:1px solid var(--line);font-size:12px}
  .badge.gpu{color:var(--green);border-color:#25543f}
  main{max-width:1180px;margin:0 auto;padding:18px 22px}
  .controls{background:var(--panel);border:1px solid var(--line);
    border-radius:12px;padding:16px;margin-bottom:16px}
  .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
  .scn{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px}
  .scn button{flex:1;min-width:200px;text-align:left;cursor:pointer;
    background:var(--panel2);border:1px solid var(--line);color:var(--ink);
    border-radius:10px;padding:11px 13px}
  .scn button.sel{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}
  .scn button b{display:block;margin-bottom:3px}
  .scn button span{color:var(--muted);font-size:12px}
  .switches{display:flex;gap:14px;flex-wrap:wrap;margin:4px 0 14px}
  .switches label{display:flex;gap:6px;align-items:center;color:var(--muted);
    font-size:13px;cursor:pointer}
  .go{cursor:pointer;border:none;border-radius:10px;padding:11px 22px;
    font-weight:600;font-size:14px}
  .go.start{background:var(--accent);color:#fff}
  .go.stop{background:#3a2530;color:var(--red);border:1px solid #5a2e37}
  .go:disabled{opacity:.45;cursor:not-allowed}
  .grid{display:grid;grid-template-columns:1.25fr 1fr;gap:16px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
    padding:16px}
  .card h3{margin:0 0 12px;font-size:13px;color:var(--muted);
    text-transform:uppercase;letter-spacing:.04em}
  #arena{width:100%;border-radius:10px;background:#000;display:block;
    aspect-ratio:1/1;object-fit:contain}
  .say{margin-top:12px;padding:11px 13px;border-radius:9px;background:var(--panel2);
    border-left:3px solid var(--accent);min-height:42px}
  .say b{color:var(--accent)}
  .stat{display:flex;justify-content:space-between;padding:7px 0;
    border-bottom:1px dashed var(--line)}
  .stat span{color:var(--muted)}
  .stat b{font-variant-numeric:tabular-nums}
  .bar{height:9px;border-radius:6px;background:var(--panel2);overflow:hidden;
    margin:5px 0 12px}
  .bar>div{height:100%;transition:width .3s}
  .barlab{display:flex;justify-content:space-between;font-size:12px;
    color:var(--muted)}
  .phase{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12px;
    background:var(--panel2);border:1px solid var(--line)}
  .phase.awake{color:var(--green)} .phase.asleep{color:var(--blue)}
  .phase.colony{color:var(--green)}
  .phase.error{color:var(--red)} .phase.done{color:var(--orange)}
  .charts{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-top:16px}
  canvas{width:100%;height:150px;background:var(--panel2);border-radius:9px}
  .ctitle{font-size:12px;color:var(--muted);margin-bottom:6px}
  @media(max-width:900px){.grid{grid-template-columns:1fr}.charts{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>🌱 Embodied Agent — live</h1>
  <span class="sub">watch a body — or a whole colony — learn to stay alive</span>
  <span id="dev" class="badge">device …</span>
</header>
<main>
  <div class="controls">
    <div class="scn" id="scn"></div>
    <div class="switches" id="switches"></div>
    <div class="row">
      <button class="go start" id="start">▶ Start</button>
      <button class="go stop" id="stop" disabled>■ Stop</button>
      <span id="msg" class="sub" style="margin-left:8px"></span>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <h3 id="worldTitle">The world</h3>
      <img id="arena" src="/api/frame.png" alt="arena">
      <div class="say" id="say">Pick a scenario above and press <b>Start</b>.</div>
    </div>
    <div class="card">
      <h3 id="panelTitle">Body &amp; learning</h3>
      <div class="barlab"><span id="e_l">Energy (food)</span><b id="e_v">—</b></div>
      <div class="bar"><div id="e_b" style="width:0;background:var(--green)"></div></div>
      <div class="barlab"><span>Temperature (comfort)</span><b id="t_v">—</b></div>
      <div class="bar"><div id="t_b" style="width:0;background:var(--blue)"></div></div>
      <div class="barlab"><span>Integrity (health)</span><b id="i_v">—</b></div>
      <div class="bar"><div id="i_b" style="width:0;background:var(--orange)"></div></div>
      <div class="stat"><span>State</span><b><span id="phase" class="phase">idle</span></b></div>
      <div class="stat"><span>Step</span><b id="step">0</b></div>
      <div class="stat"><span>Speed</span><b id="sps">0</b> <span>steps/s</span></div>
      <div id="singleStats">
        <div class="stat"><span>Food eaten (life)</span><b id="food">0</b></div>
        <div class="stat"><span>Eval reward vs random</span><b id="er">—</b></div>
      </div>
      <div id="colonyStats" style="display:none">
        <div class="stat"><span>Inspecting (click a creature)</span><b id="inspecting">—</b></div>
        <div class="stat"><span>Population alive</span><b id="pop">—</b></div>
        <div class="stat"><span>Births / Deaths</span><b><span id="births">0</span> / <span id="deaths">0</span></b></div>
        <div class="stat"><span>Newest generation</span><b id="gen">0</b></div>
      </div>
      <div class="stat"><span>Prediction error (world model)</span><b id="wm">—</b></div>
    </div>
  </div>

  <div class="card" style="margin-top:16px">
    <h3 id="sensesTitle">👁 What it senses (first person)</h3>
    <div style="display:flex;gap:18px;align-items:center;flex-wrap:wrap">
      <img id="senses" src="/api/senses.png" alt="senses"
           style="width:280px;height:280px;border-radius:10px;background:#0e1116">
      <div class="sub" id="sensesHint" style="max-width:460px"></div>
    </div>
  </div>

  <div class="charts">
    <div class="card"><div class="ctitle" id="t_reward">Reward over time</div>
      <canvas id="c_reward"></canvas></div>
    <div class="card"><div class="ctitle" id="t_wm">World-model prediction error (should fall)</div>
      <canvas id="c_wm"></canvas></div>
    <div class="card"><div class="ctitle" id="t_energy">Body energy over time</div>
      <canvas id="c_energy"></canvas></div>
  </div>
</main>

<script>
let SEL="colony", META=null;
const $=id=>document.getElementById(id);

function drawChart(cv, ys, color){
  const c=cv.getContext('2d'), W=cv.width=cv.clientWidth, H=cv.height=cv.clientHeight;
  c.clearRect(0,0,W,H);
  if(!ys||ys.length<2)return;
  let lo=Math.min(...ys), hi=Math.max(...ys); if(hi-lo<1e-6){hi+=1;lo-=1;}
  const pad=8, x=i=>pad+i*(W-2*pad)/(ys.length-1),
        y=v=>H-pad-(v-lo)*(H-2*pad)/(hi-lo);
  c.strokeStyle=color; c.lineWidth=2; c.beginPath();
  ys.forEach((v,i)=>{i?c.lineTo(x(i),y(v)):c.moveTo(x(i),y(v));});
  c.stroke();
  c.globalAlpha=.12; c.lineTo(x(ys.length-1),H-pad); c.lineTo(x(0),H-pad);
  c.closePath(); c.fillStyle=color; c.fill(); c.globalAlpha=1;
}
function bar(el,val,txt){$(el+'_b').style.width=Math.max(0,Math.min(1,val))*100+'%';
  $(el+'_v').textContent=txt;}
const num=(v,d=2)=>(v==null?'—':(+v).toFixed(d));

async function poll(){
  let s; try{s=await (await fetch('/api/state')).json();}catch(e){return;}
  if(!META){META=s; buildControls(s);}
  const dev=$('dev'); dev.textContent=(s.device==='cuda'?'GPU (CUDA)':'CPU');
  dev.className='badge'+(s.device==='cuda'?' gpu':'');
  const st=s.status||{}, colony=(s.mode==='colony');
  $('msg').textContent=st.message||'';
  $('say').innerHTML='<b>'+(st.phase||'')+'</b> — '+(st.message||'');
  $('step').textContent=st.step??0; $('sps').textContent=st.sps??0;
  const ph=$('phase'); ph.textContent=st.phase||'idle'; ph.className='phase '+(st.phase||'');
  bar('e',st.energy??0,num(st.energy)); bar('t',st.temp??0,num(st.temp));
  bar('i',st.integrity??0,num(st.integrity));
  $('wm').textContent=st.wm_loss??'—';
  // mode-specific panels
  $('singleStats').style.display=colony?'none':'';
  $('colonyStats').style.display=colony?'':'none';
  $('worldTitle').textContent=colony?'The colony (each dot is a creature, colour = generation)'
                                     :'The world (the agent is the blue dot)';
  $('panelTitle').textContent=colony?'Inspected creature & colony':'Body & learning';
  $('e_l').textContent=colony?'Energy (this creature)':'Energy (food)';
  $('t_reward').textContent=colony?'Population over time':'Reward over time (higher = healthier)';
  $('t_energy').textContent=colony?'Mean colony energy':'Body energy over time';
  if(colony){
    const f=st.focus;
    $('inspecting').textContent=f?('#'+f.id+'  ·  gen '+(f.generation+1)+'  ·  age '+f.age):'—';
    $('pop').textContent=st.population??'—'; $('births').textContent=st.births??0;
    $('deaths').textContent=st.deaths??0; $('gen').textContent=(st.generation!=null?st.generation+1:0);
  } else {
    $('food').textContent=st.food_total??0; $('er').textContent=(st.eval_reward??'—');
  }
  $('start').disabled=s.running; $('stop').disabled=!s.running;
  const h=s.history||{};
  drawChart($('c_reward'),h.reward,colony?'#39d98a':'#4c8dff');
  drawChart($('c_wm'),h.wm_loss,'#ffab4c');
  drawChart($('c_energy'),h.energy,'#39d98a');
  if(s.running){
    const t=Date.now();
    $('arena').src='/api/frame.png?t='+t;
    $('senses').src='/api/senses.png?t='+t;
  }
  // "what it sees" hint
  const legend='The fan is its <b>vision</b> (green = food, grey = wall), the '+
    'purple arrow is <b>smell</b>, red arcs are <b>touch</b>, and the corner '+
    'bars are its <b>body</b> (energy / temperature / health).';
  if(colony){
    $('arena').style.cursor='crosshair';
    const f=st.focus;
    $('sensesHint').innerHTML=(f?('Inspecting creature <b>#'+f.id+'</b> '+
      '(generation '+(f.generation+1)+', age '+f.age+', energy '+f.energy+'). '):'')+
      '<br><b>Click any creature</b> in the world above to see through its eyes.<br><br>'+legend;
  } else {
    $('arena').style.cursor='default';
    $('sensesHint').innerHTML='This is the agent’s own view.<br><br>'+legend;
  }
}

function buildControls(s){
  const scn=$('scn'); scn.innerHTML='';
  const order=['colony','bio_gpu','bio_cpu','baseline'];
  const name={colony:'🐜 Colony · community',bio_gpu:'GPU · biological',
    bio_cpu:'CPU · biological',baseline:'CPU · baseline'};
  order.forEach(k=>{ if(!(k in s.scenarios))return;
    const b=document.createElement('button'); b.dataset.k=k;
    b.innerHTML='<b>'+(name[k]||k)+'</b><span>'+s.scenarios[k]+'</span>';
    if(k===SEL)b.classList.add('sel');
    b.onclick=()=>{SEL=k;[...scn.children].forEach(c=>c.classList.remove('sel'));
      b.classList.add('sel');};
    scn.appendChild(b);
  });
  const sw=$('switches'); sw.innerHTML='<span class="sub">Biological switches:</span>';
  Object.entries(s.switches).forEach(([k,label])=>{
    const l=document.createElement('label');
    l.innerHTML='<input type="checkbox" id="sw_'+k+'" checked> '+label;
    sw.appendChild(l);
  });
}

$('start').onclick=async()=>{
  const switches={};
  ['neuromod','sleep','dev','metab'].forEach(k=>{const el=$('sw_'+k);
    if(el)switches[k]=el.checked;});
  await fetch('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({scenario:SEL,switches})});
};
$('stop').onclick=()=>fetch('/api/stop',{method:'POST'});

// click a creature in the world to inspect what it sees (colony)
$('arena').onclick=(ev)=>{
  const r=ev.target.getBoundingClientRect();
  const fx=(ev.clientX-r.left)/r.width, fy=(ev.clientY-r.top)/r.height;
  fetch('/api/focus',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({fx,fy})});
};

poll(); setInterval(poll,400);
</script>
</body>
</html>
"""
