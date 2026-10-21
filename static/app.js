(() => {
  const $ = id => document.getElementById(id);
  const video = $('video'), previewImage = $('previewImage'), empty = $('empty'), cameraTag = $('cameraTag');
  const overlay = $('overlay'), fpsTag = $('fps'), liveBtn = $('live');
  const start = $('start'), switchBtn = $('switch'), capture = $('capture'), fileInput = $('file'), analyze = $('analyze'), message = $('message');
  let stream = null, facingMode = 'user', selectedBlob = null, selectedDataUrl = null, selectedName = 'camera-capture.jpg';
  const emojis = {Anger:'😠',Contempt:'😒',Disgust:'🤢',Fear:'😨',Happy:'😄',Neutral:'😐',Sad:'😢',Surprise:'😲'};
  const say = (text, error=false) => { message.textContent=text||''; message.classList.toggle('error',error); };
  function stopCamera(){ stopLive(); video.classList.remove('mirror'); liveBtn.disabled=true; if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;} video.srcObject=null; cameraTag.hidden=true;capture.disabled=true;switchBtn.disabled=true;start.textContent='Start camera'; }
  function showImage(url){video.hidden=true;previewImage.src=url;previewImage.hidden=false;empty.hidden=true;cameraTag.hidden=true;analyze.disabled=false;}
  async function startCamera(){
    if(!navigator.mediaDevices?.getUserMedia){say('Camera access requires HTTPS (or localhost). You can choose an image instead.',true);return;}
    stopCamera();
    try{stream=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:{ideal:facingMode},width:{ideal:1280},height:{ideal:720}}});video.srcObject=stream;await video.play();selectedBlob=null;selectedDataUrl=null;video.hidden=false;previewImage.hidden=true;empty.hidden=true;cameraTag.hidden=false;capture.disabled=false;switchBtn.disabled=false;liveBtn.disabled=false;video.classList.toggle('mirror',facingMode==='user');start.textContent='Restart camera';analyze.disabled=true;say('Camera ready. Start live detect, or capture a frame to analyze it.');}
    catch(e){stopCamera();say(e.name==='NotAllowedError'?'Camera permission denied. Allow camera access or upload a photo.':`Could not start camera: ${e.message||e}. Try HTTPS or upload a photo.`,true);}
  }
  start.addEventListener('click',startCamera);
  switchBtn.addEventListener('click',()=>{facingMode=facingMode==='user'?'environment':'user';startCamera();});
  capture.addEventListener('click',()=>{if(!video.videoWidth){say('Camera is still starting. Try again.',true);return;}const c=document.createElement('canvas');const scale=Math.min(1,1600/Math.max(video.videoWidth,video.videoHeight));c.width=Math.round(video.videoWidth*scale);c.height=Math.round(video.videoHeight*scale);const ctx=c.getContext('2d');if(facingMode==='user'){ctx.translate(c.width,0);ctx.scale(-1,1);}ctx.drawImage(video,0,0,c.width,c.height);selectedDataUrl=c.toDataURL('image/jpeg',.9);selectedBlob=null;selectedName='camera-capture.jpg';showImage(selectedDataUrl);stopCamera();say('Frame captured. Press Analyze expression.');});
  fileInput.addEventListener('change',()=>{const f=fileInput.files?.[0];if(!f)return;if(!f.type.startsWith('image/')){say('Choose an image file.',true);return;}if(f.size>10*1024*1024){say('Image exceeds 10 MB. Choose a smaller image.',true);return;}stopCamera();selectedBlob=f;selectedDataUrl=null;selectedName=f.name;const r=new FileReader();r.onload=()=>{showImage(r.result);say('Image ready. Press Analyze expression.');};r.onerror=()=>say('Could not read that image.',true);r.readAsDataURL(f);});
  function render(data){$('emptyResult').hidden=true;$('result').hidden=false;$('emoji').textContent=emojis[data.emotion]||'🙂';$('emotion').textContent=data.emotion;const conf=Number(data.confidence||0);$('confidence').textContent=`${(conf*100).toFixed(1)}% model score`;$('confidenceBar').style.width=`${Math.max(0,Math.min(100,conf*100))}%`;const list=$('scores');list.replaceChildren();Object.entries(data.probabilities||{}).sort((a,b)=>b[1]-a[1]).forEach(([name,value])=>{const row=document.createElement('div');row.className='scoreRow';const label=document.createElement('span');label.className='name';label.textContent=name;const track=document.createElement('div');track.className='miniTrack';const fill=document.createElement('span');fill.style.width=`${Math.max(0,Math.min(100,value*100))}%`;if(name===data.emotion)fill.style.background='var(--accent)';track.appendChild(fill);const val=document.createElement('span');val.className='value';val.textContent=`${(value*100).toFixed(1)}%`;row.append(label,track,val);list.appendChild(row);});}
  analyze.addEventListener('click',async()=>{if(!selectedBlob&&!selectedDataUrl){say('Capture a frame or choose an image first.',true);return;}analyze.disabled=true;const old=analyze.innerHTML;analyze.textContent='Analyzing…';say('Running prediction…');try{let response;if(selectedBlob){const form=new FormData();form.append('image',selectedBlob,selectedName);response=await fetch('/api/predict',{method:'POST',body:form});}else{response=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:selectedDataUrl})});}const data=await response.json();if(!response.ok)throw new Error(data.error||`Request failed (${response.status})`);render(data);say('Prediction complete. These are model scores, not a definite reading of emotion.');}catch(e){say(e.message||'Prediction failed. Check Flask and the model file.',true);}finally{analyze.disabled=!(selectedBlob||selectedDataUrl);analyze.innerHTML=old;}});

  // ---- Live detection: stream downscaled frames, draw a box + label per face ----
  let live = false, liveTimer = null, frames = 0, fpsStart = 0;
  const grab = document.createElement('canvas');
  function stopLive(){
    live=false; clearTimeout(liveTimer); liveTimer=null;
    overlay.hidden=true; overlay.getContext('2d').clearRect(0,0,overlay.width,overlay.height);
    fpsTag.hidden=true; liveBtn.textContent='Start live detect'; liveBtn.classList.remove('live');
  }
  function drawFaces(res){
    const rect=overlay.getBoundingClientRect(), dpr=window.devicePixelRatio||1;
    overlay.width=Math.round(rect.width*dpr); overlay.height=Math.round(rect.height*dpr);
    const ctx=overlay.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,rect.width,rect.height);
    // Map frame coords into the object-fit:contain area of the video element.
    const s=Math.min(rect.width/res.width,rect.height/res.height), ox=(rect.width-res.width*s)/2, oy=(rect.height-res.height*s)/2;
    const mirrored=video.classList.contains('mirror');
    ctx.font='600 13px system-ui,sans-serif'; ctx.lineWidth=2.5;
    res.faces.forEach((f,i)=>{
      let [x,y,w,h]=f.box; if(mirrored) x=res.width-x-w;
      const X=ox+x*s, Y=oy+y*s, Wd=w*s, Ht=h*s, color=i===0?'#a8f36a':'#8fb8ff';
      ctx.strokeStyle=color; ctx.strokeRect(X,Y,Wd,Ht);
      const label=`${emojis[f.emotion]||''} ${f.emotion} ${(f.confidence*100).toFixed(0)}%`, tw=ctx.measureText(label).width+12;
      const ly=Y>24?Y-24:Y+Ht; ctx.fillStyle=color; ctx.fillRect(X-1,ly,tw,22); ctx.fillStyle='#0b1120'; ctx.fillText(label,X+5,ly+15);
    });
  }
  async function liveStep(){
    if(!live) return;
    if(!video.videoWidth){ liveTimer=setTimeout(liveStep,200); return; }
    const scale=Math.min(1,640/Math.max(video.videoWidth,video.videoHeight));
    grab.width=Math.round(video.videoWidth*scale); grab.height=Math.round(video.videoHeight*scale);
    grab.getContext('2d').drawImage(video,0,0,grab.width,grab.height);
    try{
      const r=await fetch('/api/predict_live',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:grab.toDataURL('image/jpeg',.8)})});
      const data=await r.json(); if(!r.ok) throw new Error(data.error||`Request failed (${r.status})`);
      if(!live) return;
      drawFaces(data);
      if(data.faces.length){ render(data.faces[0]); say(`${data.faces.length} face${data.faces.length>1?'s':''} detected · live model scores, not a definite reading of emotion.`); }
      else say('No face detected — face the camera with good lighting.');
      frames++; const secs=(performance.now()-fpsStart)/1000; if(secs>1){ fpsTag.textContent=`LIVE · ${(frames/secs).toFixed(1)} FPS`; frames=0; fpsStart=performance.now(); }
    }catch(e){ if(live){ say(e.message||'Live prediction failed.',true); stopLive(); } return; }
    liveTimer=setTimeout(liveStep,60);
  }
  liveBtn.addEventListener('click',()=>{
    if(live){ stopLive(); say('Live detection stopped.'); return; }
    live=true; frames=0; fpsStart=performance.now(); overlay.hidden=false; fpsTag.hidden=false; fpsTag.textContent='LIVE';
    liveBtn.textContent='Stop live detect'; liveBtn.classList.add('live'); say('Live detection running…'); liveStep();
  });
  window.addEventListener('pagehide',stopCamera);
  fetch('/health').then(r=>r.json()).then(d=>{if(!d.model_ready){$('modelBadge').textContent='Model not loaded';$('modelBadge').classList.remove('ready');$('modelBadge').classList.add('offline');say(d.error||'Model checkpoint was not found.',true);}}).catch(()=>{});
})();
