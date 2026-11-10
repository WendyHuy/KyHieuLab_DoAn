"use strict";
const $ = id => document.getElementById(id);
const state = {
  config:null, blob:null, objectUrl:null, blobId:null, videoName:"", source:"",
  stream:null, recorder:null, timer:null, recording:false, cameraPending:false,
  cameraToken:0, sourceToken:0, revision:0, busy:false, loadingSample:false,
  prediction:null, trialSaved:false, savingTrial:false, activeTab:"recognize",
};
const TAB_NAMES = {recognize:"Nhận diện", experiment:"Thực nghiệm", evaluation:"Đánh giá", guide:"Hướng dẫn"};
const TAB_HASHES = {recognize:"#nhan-dien",experiment:"#thuc-nghiem",evaluation:"#danh-gia",guide:"#huong-dan"};
const MODEL_NAMES = {tcn:"TCN gốc",tcn_augmented:"TCN tăng cường",mlp:"MLP",tcn_reject:"TCN có từ chối"};
const displayNumber = (value, digits=0) => Number.isFinite(Number(value)) ? Number(value).toLocaleString("vi-VN",{minimumFractionDigits:digits,maximumFractionDigits:digits}) : "—";
const displayPercent = value => value === null || value === undefined ? "—" : displayNumber(Number(value)*100,1)+"%";
const status = (text, isError=false) => {$("status").textContent=text;$("status").classList.toggle("error",isError);};

function setTab(name, focus=false) {
  if (!Object.hasOwn(TAB_NAMES,name)) return;
  if (state.recording) {status("Đợi đoạn quay kết thúc rồi chuyển phần nhé.");return;}
  if (name !== "recognize" && (state.stream || state.cameraPending)) stopCamera();
  state.activeTab=name;
  for (const key of Object.keys(TAB_NAMES)) {
    const active=key===name;
    $("tab-"+key).setAttribute("aria-selected",String(active));
    $("tab-"+key).classList.toggle("active",active);
    $("tab-"+key).tabIndex=active?0:-1;
    $("panel-"+key).hidden=!active;
  }
  $("current-view").textContent=TAB_NAMES[name];
  history.replaceState(null,"",TAB_HASHES[name]);
  if (focus) $("tab-"+name).focus();
  if (name === "experiment") loadTrialSummary();
}
for (const name of Object.keys(TAB_NAMES)) {
  $("tab-"+name).addEventListener("click",()=>setTab(name));
  $("tab-"+name).addEventListener("keydown",event=>{
    const keys=Object.keys(TAB_NAMES), index=keys.indexOf(name);
    let next;
    if (["ArrowRight","ArrowDown"].includes(event.key)) next=(index+1)%keys.length;
    else if (["ArrowLeft","ArrowUp"].includes(event.key)) next=(index+keys.length-1)%keys.length;
    else if (event.key==="Home") next=0;
    else if (event.key==="End") next=keys.length-1;
    else return;
    event.preventDefault();setTab(keys[next],true);
  });
}
document.querySelector(".brand").addEventListener("click",event=>{event.preventDefault();setTab("recognize");});
$("experiment-return").addEventListener("click",()=>setTab("recognize"));
$("go-experiment").addEventListener("click",()=>{setTab("experiment");if(state.prediction) $("signer").focus();});

function predictionMessage(answer) {
  if(answer?.rejected_as_outside || /^(?:__OTHER__|OTHER)$/.test(String(answer?.message||"")))return "Có thể nằm ngoài bộ 5 ký hiệu";
  return answer?.message||"Chưa chắc chắn";
}

function renderTrialCurrent() {
  const answer=state.prediction;
  $("trial-prediction").textContent=answer ? predictionMessage(answer) : "Chưa có kết quả";
  if (answer) {
    const accepted=answer.rejected_as_outside ? "Có thể ngoài bộ ký hiệu" : answer.recognized ? "Qua ngưỡng gợi ý" : "Cần người kiểm tra";
    $("trial-context").textContent=(MODEL_NAMES[answer.model]||answer.model)+" · "+accepted+" · "+state.videoName;
  } else $("trial-context").textContent="Hãy nhận diện một video trước khi ghi phiếu.";
  $("save-trial").disabled=!answer||state.trialSaved||state.savingTrial;
  $("save-trial").textContent=state.savingTrial ? "Đang ghi phiếu…" : state.trialSaved ? "Đã ghi phiếu cho lượt này" : "Ghi phiếu kết quả";
}

function invalidate() {
  state.revision++;
  state.prediction=null;
  state.trialSaved=false;
  $("trial-status").textContent="";
  $("actual-label").value="";
  $("actual-label-note").textContent=state.source==="outside-sample"?"Mẫu đang chọn nằm ngoài 5 nhãn. Bạn có thể chọn Ngoài bộ 5 ký hiệu sau khi đối chiếu; đây vẫn là phản hồi chưa được duyệt.":"Dùng Ngoài bộ 5 ký hiệu khi video không thuộc các nhãn đang hỗ trợ.";
  $("result").textContent=state.blob ? "Sẵn sàng nhận diện" : "Chờ video";
  $("result").classList.remove("uncertain","outside");
  $("result-state").textContent="Chờ đầu vào";
  $("result-state").className="result-state";
  $("result-detail").textContent=state.blob ? "Chọn trọn đoạn ký và bấm nhận diện để xem gợi ý." : "Kết quả sẽ xuất hiện sau khi bạn bấm nhận diện.";
  $("top").replaceChildren();
  $("go-experiment").disabled=true;
  $("step-video").classList.add("active");$("step-video").classList.remove("complete");
  $("step-infer").classList.toggle("active",Boolean(state.blob));
  $("step-infer").classList.remove("complete");$("step-record").classList.remove("active");
  renderTrialCurrent();
}

function controls() {
  const locked=state.busy||state.recording||state.loadingSample||state.cameraPending||!state.config;
  for (const id of ["file","examples","start","end","model","camera"]) $(id).disabled=locked;
  $("upload-label").classList.toggle("disabled",locked);
  $("upload-label").setAttribute("aria-disabled",String(locked));
  $("camera").disabled=locked||Boolean(state.stream);
  $("camera-label").textContent=state.cameraPending ? "Đang mở camera…" : state.stream ? "Camera đã bật" : "Bật camera";
  $("record").disabled=locked||!state.stream;
  $("stop-camera").disabled=state.recording||(!state.stream&&!state.cameraPending);
  $("predict").disabled=locked||!state.blob;
  $("download-video").disabled=locked||!state.blob;
  $("reset-crop").disabled=locked||!state.blob;
  $("predict-label").textContent=state.busy?"Đang nhận diện…":"Nhận diện đoạn ký";
  $("panel-recognize").classList.toggle("is-busy",state.busy);
  $("video-title").setAttribute("aria-busy",String(state.busy));
  for(const name of Object.keys(TAB_NAMES)) $("tab-"+name).disabled=state.recording;
  renderTrialCurrent();
}

function stopCamera() {
  state.cameraToken++;
  state.stream?.getTracks().forEach(track=>track.stop());
  state.stream=null;state.cameraPending=false;
  if ($("video").srcObject) $("video").srcObject=null;
  if (!state.blob) $("video-empty").hidden=false;
  controls();
}

function releaseBlob() {
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.blob=null;state.objectUrl=null;
  $("video").removeAttribute("src");
}

function preview(blob, name="Video vừa quay", source="camera") {
  stopCamera();releaseBlob();
  state.blob=blob;state.blobId=crypto.randomUUID();state.videoName=name;state.source=source;
  state.objectUrl=URL.createObjectURL(blob);
  $("video").srcObject=null;$("video").src=state.objectUrl;$("video").muted=false;
  $("video-empty").hidden=true;
  $("source-badge").textContent=source==="outside-sample"?"Ngoài bộ 5 ký hiệu":source==="sample"?"Clip mẫu đã học":source==="upload"?"Video tải lên":"Video vừa quay";
  $("source-badge").classList.toggle("outside-sample",source==="outside-sample");
  $("source-badge").title=name;
  $("source-note").textContent=source==="outside-sample"?"Mẫu ngoài 5 ký hiệu từ bộ dữ liệu cũ; ứng viên TCN có từ chối đã dùng mẫu này để học. Dùng để demo, không tính là kiểm tra mới.":source==="sample"?"Đây là clip trong phần học; kết quả chỉ kiểm tra khả năng chạy demo.":"Đang chọn: "+name;
  $("start").value=0;$("end").value=0;
  $("start").removeAttribute("max");$("end").removeAttribute("max");
  invalidate();controls();
  status(source==="outside-sample"?"Đã chọn mẫu ngoài bộ 5 ký hiệu. Bạn có thể so sánh các mô hình; đây là mẫu phát triển đã học bởi ứng viên.":"Đã có video. Chọn đoạn ký rồi bấm nhận diện.");
}

$("file").addEventListener("change",()=>{
  const file=$("file").files[0];if(!file)return;
  if(!file.size) {status("Video đang chọn không có dữ liệu. Chọn một tệp khác.",true);$("file").value="";return;}
  if(file.size>64*1024*1024) {status("Video quá lớn. Chọn clip dưới 64 MB.",true);$("file").value="";return;}
  state.sourceToken++;
  $("examples").value="";
  preview(file,file.name,"upload");
});

$("examples").addEventListener("change",async()=>{
  const selection=$("examples").value;if(!selection)return;
  const outside=selection.startsWith("outside:"),id=outside?selection.slice(8):selection;
  const sourceToken=++state.sourceToken;
  invalidate();state.loadingSample=true;controls();status("Đang mở clip mẫu…");
  try {
    const samples=outside?(state.config.outside_samples||[]):state.config.samples;
    const sample=samples.find(row=>row.id===id);
    if(!sample)throw Error("Mẫu đang chọn không nằm trong danh sách. Hãy chọn lại.");
    const response=await fetch((outside?"/outside-samples/":"/samples/")+encodeURIComponent(id));
    if(!response.ok) throw Error("Không tải được clip mẫu. Hãy chọn lại hoặc tải video của bạn.");
    const blob=await response.blob();
    if(sourceToken!==state.sourceToken)return;
    $("file").value="";
    preview(blob,(outside?"Ngoài bộ 5 · ":"")+(sample.label||"Clip mẫu")+" · "+id,outside?"outside-sample":"sample");
  } catch(error) {if(sourceToken===state.sourceToken)status(error.message,true);}
  finally {if(sourceToken===state.sourceToken){state.loadingSample=false;controls();}}
});

for(const id of ["model","start","end"]) $(id).addEventListener("input",()=>{
  invalidate();status("Đã đổi cấu hình. Nhận diện lại để ghi phiếu đúng lượt.");
});
$("reset-crop").addEventListener("click",()=>{
  $("start").value=0;$("end").value=0;invalidate();status("Đã chọn toàn bộ clip. Nếu video dài, hãy chọn một đoạn ký tối đa 15 giây.");
});
$("video").addEventListener("loadedmetadata",()=>{
  if(!state.blob)return;
  const duration=$("video").duration;
  if(Number.isFinite(duration)&&duration>0) {
    $("start").max=String(duration);$("end").max=String(duration);
    $("crop-note").textContent="Video dài "+displayNumber(duration,1)+" giây. Cuối = 0 lấy đến hết; chọn trọn một động tác, tối đa 15 giây.";
  } else $("crop-note").textContent="Cuối = 0 lấy đến hết. Chọn trọn một động tác, tối đa 15 giây.";
});
$("stop-camera").addEventListener("click",()=>{stopCamera();$("camera-status").textContent="Camera đã tắt.";status("Đã tắt camera.");});

$("camera").addEventListener("click",async()=>{
  stopCamera();releaseBlob();state.sourceToken++;
  $("examples").value="";$("file").value="";
  state.videoName="Camera";state.source="camera";
  $("source-badge").textContent="Đang mở camera";$("source-badge").classList.remove("outside-sample");
  $("source-note").textContent="Video mới sẽ được xử lý sau khi quay xong.";invalidate();
  state.cameraPending=true;const token=++state.cameraToken;controls();
  $("camera-status").textContent="Cho phép trình duyệt sử dụng camera để bắt đầu.";
  status("Đang chờ quyền camera…");
  try {
    if(!navigator.mediaDevices?.getUserMedia)throw Error("Trình duyệt chưa hỗ trợ camera tại địa chỉ này.");
    const stream=await navigator.mediaDevices.getUserMedia({video:{width:1280,height:720},audio:false});
    if(token!==state.cameraToken){stream.getTracks().forEach(track=>track.stop());return;}
    state.stream=stream;state.cameraPending=false;
    $("video").srcObject=stream;$("video").muted=true;$("video-empty").hidden=true;
    $("source-badge").textContent="Camera trực tiếp";
    controls();await $("video").play();
    if(token!==state.cameraToken)return;
    $("camera-status").textContent="Camera sẵn sàng. Thực hiện một động tác ngay sau khi bấm quay.";
    status("Bấm Quay 5 giây để bắt đầu thu video.");
  } catch(error) {
    if(token!==state.cameraToken)return;
    stopCamera();$("source-badge").textContent="Chưa có video";$("camera-status").textContent="Bạn có thể tải video MP4/WebM có sẵn thay thế.";
    status("Không mở được camera: "+error.message,true);
  }
});

$("record").addEventListener("click",()=>{
  if(!state.stream||state.recording)return;
  const stream=state.stream;
  try {
    if(typeof MediaRecorder==="undefined")throw Error("Trình duyệt chưa hỗ trợ quay. Hãy dùng chức năng tải video.");
    const mime=["video/webm;codecs=vp9","video/webm;codecs=vp8","video/webm","video/mp4"].find(type=>MediaRecorder.isTypeSupported(type));
    const recorder=new MediaRecorder(stream,mime?{mimeType:mime}:{});
    const chunks=[];let failed=false;
    state.recorder=recorder;
    recorder.ondataavailable=event=>{if(event.data.size)chunks.push(event.data);};
    recorder.onerror=()=>{
      failed=true;clearInterval(state.timer);state.recording=false;state.recorder=null;
      $("recording-pill").hidden=true;stopCamera();controls();
      status("Không quay được video. Thử lại hoặc tải một clip có sẵn.",true);
    };
    recorder.onstop=()=>{
      clearInterval(state.timer);state.recording=false;state.recorder=null;$("recording-pill").hidden=true;
      if(failed)return;
      if(chunks.length)preview(new Blob(chunks,{type:recorder.mimeType}),"Đoạn quay 5 giây","camera");
      else {stopCamera();status("Chưa thu được video. Thử quay lại.",true);}
      $("camera-status").textContent="Đã quay xong. Camera đã tắt; xem lại trước khi nhận diện.";
      controls();
    };
    state.recording=true;invalidate();controls();recorder.start();
    let remaining=5;$("recording-pill").hidden=false;$("recording-time").textContent="Đang quay · 5 giây";
    $("camera-status").textContent="Đang quay. Giữ tay trong khung và thực hiện một ký hiệu.";
    status("Đang quay đoạn ký…");
    state.timer=setInterval(()=>{
      remaining--;$("recording-time").textContent="Đang quay · "+remaining+" giây";
      if(remaining<=0){clearInterval(state.timer);if(recorder.state==="recording")recorder.stop();}
    },1000);
  } catch(error) {
    clearInterval(state.timer);state.recording=false;state.recorder=null;$("recording-pill").hidden=true;
    stopCamera();controls();status(error.message,true);
  }
});

function renderPrediction(answer) {
  const outside=Boolean(answer.rejected_as_outside);
  $("result").textContent=predictionMessage(answer);
  $("result").classList.toggle("uncertain",!answer.recognized);
  $("result").classList.toggle("outside",outside);
  $("result-state").textContent=outside?"Ngoài bộ ký hiệu":answer.recognized?"Qua ngưỡng gợi ý":"Cần xác nhận";
  $("result-state").className="result-state "+(outside?"outside":answer.recognized?"accepted":"uncertain");
  $("result-detail").textContent="Xử lý "+displayNumber(answer.elapsed_seconds,2)+" giây · thấy tay ở "+displayNumber((answer.hand_fraction||0)*100)+"% bước lấy mẫu."+(outside?" Chưa có gợi ý được chấp nhận; hãy xem lại hoặc ghi phiếu theo nhãn thực tế.":"");
  $("top").replaceChildren();
  for(const item of answer.top||[]) {
    if(!state.config?.labels?.includes(item.label))continue;
    const row=document.createElement("div");row.className="score";
    const caption=document.createElement("div");caption.className="score-caption";
    const label=document.createElement("span");label.textContent=item.label;
    const score=document.createElement("span");score.textContent="điểm "+displayNumber(item.score,3);
    const progress=document.createElement("progress");progress.max=1;progress.value=Number(item.score)||0;progress.setAttribute("aria-label",item.label+" điểm mô hình");
    caption.append(label,score);row.append(caption,progress);$("top").append(row);
  }
  $("go-experiment").disabled=false;
  $("step-video").classList.remove("active");$("step-video").classList.add("complete");
  $("step-infer").classList.remove("active");$("step-infer").classList.add("complete");$("step-record").classList.add("active");
  renderTrialCurrent();
}

$("predict").addEventListener("click",async()=>{
  if(!state.blob||state.busy||state.loadingSample||state.recording||state.cameraPending)return;
  const start=Number($("start").value),end=Number($("end").value),duration=$("video").duration;
  if(!Number.isFinite(start)||!Number.isFinite(end)||start<0||end<0||(end>0&&end<=start))return status("Mốc kết thúc phải lớn hơn mốc bắt đầu. Cuối = 0 lấy đến hết clip.",true);
  if(end>0&&end-start>15)return status("Chọn trọn một động tác, tối đa 15 giây.",true);
  if(Number.isFinite(duration)&&duration>0) {
    if(start>=duration)return status("Mốc bắt đầu phải nằm trong video.",true);
    if(end>duration+.15)return status("Mốc kết thúc đang vượt thời lượng video.",true);
    if(end===0&&duration-start>15.1)return status("Video dài hơn 15 giây. Chọn mốc kết thúc cho một đoạn ký ngắn.",true);
  }
  invalidate();const revision=state.revision;
  const model=$("model").value,blob=state.blob;
  state.busy=true;controls();$("result").textContent="Đang nhận diện…";
  $("result-state").textContent="Đang xử lý";status("Đang trích điểm tay và chạy mô hình trên máy này…");
  try {
    const query=new URLSearchParams({model,start,end});
    const response=await fetch("/api/predict?"+query,{method:"POST",body:blob});
    const answer=await response.json();
    if(!response.ok)throw Error(answer.error||"Nhận diện thất bại. Thử một clip MP4 ngắn hơn.");
    if(revision!==state.revision)return;
    state.prediction=answer;renderPrediction(answer);
    status(answer.rejected_as_outside?"Đây là trạng thái từ chối của mô hình. Bạn vẫn có thể ghi phiếu theo nhãn thực tế.":"Đã có gợi ý trong bộ 5 ký hiệu. Máy có thể gán nhầm động tác ngoài bộ từ; cần người kiểm tra.");
  } catch(error) {
    if(revision!==state.revision)return;
    $("result").textContent="Chưa xử lý được";$("result-state").textContent="Thử lại";
    $("result-detail").textContent="Thử một đoạn quay rõ tay hoặc chọn lại mốc cắt.";status(error.message,true);
  } finally {state.busy=false;controls();}
});

$("trial-form").addEventListener("submit",async event=>{
  event.preventDefault();const answer=state.prediction;
  if(!answer||state.savingTrial||state.trialSaved)return;
  if(!$("trial-form").checkValidity()){ $("trial-form").reportValidity();return; }
  state.savingTrial=true;renderTrialCurrent();
  try {
    const payload={inference_id:answer.inference_id,signer_code:$("signer").value.trim(),session_code:$("session").value.trim(),actual_label:$("actual-label").value};
    const response=await fetch("/api/trials",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    const data=await response.json();
    if(!response.ok)throw Error(data.error||"Chưa ghi được phiếu. Thử lại.");
    if(state.prediction===answer){state.trialSaved=true;$("trial-status").textContent="Đã ghi phiếu cho lượt này. Bạn có thể tải CSV hoặc thử một video mới.";}
    try {localStorage.setItem("khl-signer",payload.signer_code);localStorage.setItem("khl-session",payload.session_code);}catch{}
    await loadTrialSummary();
  } catch(error) {
    if(state.prediction===answer)$("trial-status").textContent=error.message;
  } finally {state.savingTrial=false;renderTrialCurrent();}
});

$("download-video").addEventListener("click",()=>{
  if(!state.objectUrl||!state.blob)return;
  const link=document.createElement("a");link.href=state.objectUrl;
  link.download="KHL_"+(state.prediction?.inference_id||state.blobId)+(state.blob.type.includes("mp4")?".mp4":".webm");
  document.body.append(link);link.click();link.remove();
});

function tableRow(values, labels=[]) {
  const row=document.createElement("tr");
  for(const [index,value] of values.entries()){const cell=document.createElement("td");cell.textContent=String(value);if(labels[index])cell.dataset.label=labels[index];row.append(cell);}
  return row;
}

async function loadEvaluation() {
  try {
    const response=await fetch("/api/evaluation");if(!response.ok)throw Error();
    const data=await response.json();$("metrics").replaceChildren();
    for(const model of data.models||[]) {
      const primary=model.primary,n=primary.n_unique_clips||5;
      $("metrics").append(tableRow([model.name,primary.top1_correct+"/"+n,primary.accepted_correct+"/"+n,primary.accepted_wrong],["Mô hình","Gợi ý đầu đúng","Chấp nhận đúng","Chấp nhận sai"]));
    }
    $("seed-badge").textContent="Seed "+data.primary_seed;
    $("metric-note").textContent="Mỗi mô hình được học riêng cho từng clip giữ ngoài phần học. Bảng hiển thị seed chính "+data.primary_seed+"; điểm gợi ý và lượt được chấp nhận là hai chỉ số khác nhau.";
  } catch {$("metric-note").textContent="Chưa tải được kết quả đánh giá. Thử tải lại trang hoặc mở báo cáo.";}
}

async function loadExtraEvaluation() {
  try {
    const response=await fetch("/api/evaluation-v5");if(!response.ok)throw Error();
    const data=await response.json(),outside=data.outside_vocabulary;
    if(!outside||!Array.isArray(outside.models))throw Error();
    $("outside-models").replaceChildren();
    for(const model of outside.models)$("outside-models").append(tableRow([MODEL_NAMES[model.id]||model.name||model.id,model.n??outside.n,model.rejected,model.false_accepts+" ("+displayPercent(model.false_accept_rate)+")"],["Mô hình","Video thăm dò","Đã từ chối","Chấp nhận nhầm"]));
    $("outside-badge").textContent=outside.n+" clip ngoài bộ từ";
    $("outside-note").textContent="Bảng này dùng trọng số cơ sở học đủ 8 video, tách khỏi phép so sánh theo lượt giữ mẫu v6 bên dưới. Các clip ngoài danh sách lấy từ bộ dữ liệu cũ; đây là phép thăm dò hồi cứu, chưa đại diện cho thực tế.";
    $("outside-table-wrap").hidden=false;
    $("readiness-note").hidden=false;
    $("readiness-note").textContent="Phù hợp để demo và ghi thực nghiệm có giám sát. Chưa kiểm chứng đầy đủ 5 nhãn hoặc người ký mới; chưa đủ bằng chứng để dùng nhận diện công khai.";
  } catch {
    $("outside-badge").textContent="Hồ sơ bổ sung";
    $("outside-note").textContent="Chưa mở được hồ sơ thăm dò ngoài bộ từ. Bạn vẫn có thể nhận diện và đọc kết quả giữ mẫu riêng bên trên.";
  }
}

function rejectionRow(method, result, suffix="") {
  const known=result.known,outside=result.outside;
  return tableRow([
    (MODEL_NAMES[method.id]||method.name||method.id)+suffix,
    known.accepted_correct+"/"+known.n,
    known.false_rejects+"/"+known.n,
    known.accepted_wrong+"/"+known.n,
    outside.false_accepts+"/"+outside.n+" ("+displayPercent(outside.false_accept_rate)+")"
  ],["Mô hình","Chấp nhận đúng","Từ chối mẫu hợp lệ","Chấp nhận sai","Nhận nhầm ngoài bộ từ"]);
}

async function loadRejectionEvaluation() {
  try {
    const response=await fetch("/api/rejection-v6");if(!response.ok)throw Error();
    const data=await response.json();
    if(!Array.isArray(data.methods)||!data.methods.length||!data.scope)throw Error();
    $("rejection-v6-models").replaceChildren();$("rejection-v6-seed-models").replaceChildren();
    for(const method of data.methods){
      if(!method.primary?.known||!method.primary?.outside)continue;
      $("rejection-v6-models").append(rejectionRow(method,method.primary));
      for(const result of method.by_seed||[])if(result.known&&result.outside)$("rejection-v6-seed-models").append(rejectionRow(method,result," · "+result.seed));
    }
    const scope=data.scope;
    $("rejection-v6-badge").textContent="Seed "+data.primary_seed;
    $("rejection-v6-note").textContent="Hai phương pháp dùng cùng cách giữ riêng theo video nguồn: "+scope.known_holdout_unique_clips+" clip thuộc hai nhãn được hỗ trợ và "+scope.outside_holdout_unique_clips+" clip ngoài bộ 5 ký hiệu. Từ chối mẫu hợp lệ nghĩa là máy không chấp nhận một clip có nhãn trong bộ từ. Đây là đánh giá phát triển hồi cứu trên dữ liệu đã có.";
    $("rejection-v6-wrap").hidden=false;
    $("rejection-v6-seeds").hidden=!$("rejection-v6-seed-models").children.length;
    const recommended=MODEL_NAMES[data.recommended_default]||data.recommended_default;
    $("rejection-v6-decision").textContent=data.promotion_passed?"Theo tiêu chí đã chốt cho đợt này, "+recommended+" được đề xuất làm mặc định cho demo có giám sát. Kết quả còn giới hạn theo bộ dữ liệu hiện có; chưa kiểm chứng người ký mới hoặc đủ cả 5 nhãn.":"Cấu hình đề xuất tiếp tục là "+recommended+". Mô hình có từ chối được giữ để so sánh; đợt này chưa đáp ứng đủ tiêu chí chuyển mặc định. Kết quả vẫn cần người kiểm tra.";
    $("rejection-v6-decision").hidden=false;
  } catch {
    $("rejection-v6-badge").textContent="Hồ sơ so sánh v6";
    $("rejection-v6-note").textContent="Hồ sơ so sánh mô hình có từ chối chưa sẵn sàng. Bạn vẫn có thể dùng các mô hình được máy chủ liệt kê và đọc đánh giá cơ sở bên trên.";
  }
}

let summaryRequest=0;
async function loadTrialSummary() {
  const request=++summaryRequest;
  try {
    const response=await fetch("/api/trial-summary");if(!response.ok)throw Error();
    const data=await response.json();
    if(request!==summaryRequest)return;
    $("trial-count").textContent=displayNumber(data.attempts||0);
    $("trial-videos").textContent=displayNumber(data.unique_videos||0);
    $("trial-signers").textContent=displayNumber(Array.isArray(data.signers)?data.signers.length:data.signers||0);
    $("trial-empty").hidden=Boolean(data.attempts);
    $("trial-summary-wrap").hidden=!data.attempts;
    $("trial-summary-models").replaceChildren();
    for(const model of data.models||[])$("trial-summary-models").append(tableRow([model.name||MODEL_NAMES[model.id]||model.id,model.attempts,displayPercent(model.top1_accuracy_in_vocabulary),displayPercent(model.coverage_in_vocabulary)],["Mô hình","Lượt đã ghi","Đúng theo phiếu","Chấp nhận"]));
    $("summary-status").textContent=data.training_replay_attempts?displayNumber(data.training_replay_attempts)+" lượt dùng lại video trong phần học. Xem riêng chúng khi phân tích thực nghiệm.":"Phiếu phản hồi do người kiểm tra ghi nhận, chưa được duyệt độc lập.";
  } catch {if(request===summaryRequest)$("summary-status").textContent="Chưa tải được nhật ký tổng hợp. Các phiếu đã lưu vẫn có thể xuất bằng nút CSV.";}
}
$("refresh-summary").addEventListener("click",loadTrialSummary);

async function initialize() {
  controls();
  try {
    const response=await fetch("/api/config");if(!response.ok)throw Error("Không tải được cấu hình. Kiểm tra máy chủ rồi tải lại trang.");
    const config=await response.json();
    const models=Object.entries(config.models||{}).filter(([id,name])=>id&&typeof name==="string");
    if(!models.length||!Array.isArray(config.labels)||!Array.isArray(config.samples))throw Error("Cấu hình mô hình chưa đầy đủ. Kiểm tra máy chủ rồi tải lại trang.");
    state.config=config;
    $("model").replaceChildren();
    const defaultModel=models.some(([id])=>id===config.default_model)?config.default_model:models[0][0];
    for(const [id,name] of models){MODEL_NAMES[id]=name;$("model").add(new Option(name+(id===defaultModel?" · mặc định":""),id));}
    $("label-list").replaceChildren();
    for(const label of config.labels) {
      const chip=document.createElement("span");chip.className="chip";chip.textContent=label;$("label-list").append(chip);
      $("actual-label").add(new Option(label,label));
    }
    $("actual-label").add(new Option("Ngoài bộ 5 ký hiệu",config.outside_label));
    $("examples").replaceChildren(new Option("Chọn một video để thử…",""));
    const knownGroup=document.createElement("optgroup");knownGroup.label="Trong bộ 5 ký hiệu · mẫu đã học";
    for(const row of config.samples)knownGroup.append(new Option(row.label+" · "+row.id,row.id));
    $("examples").append(knownGroup);
    if(Array.isArray(config.outside_samples)&&config.outside_samples.length){
      const outsideGroup=document.createElement("optgroup");outsideGroup.label="Ngoài 5 ký hiệu · mẫu ứng viên đã học";
      for(const row of config.outside_samples)outsideGroup.append(new Option("Ngoài bộ từ: "+row.label+" · mẫu đã học","outside:"+row.id));
      $("examples").append(outsideGroup);
    }
    $("model").value=defaultModel;
    $("connection").classList.add("ready");$("connection-text").textContent="Máy chủ sẵn sàng";
    try {$("signer").value=localStorage.getItem("khl-signer")||"";$("session").value=localStorage.getItem("khl-session")||"";}catch{}
    controls();status("Sẵn sàng. Chọn clip mẫu, tải video hoặc bật camera.");
  } catch(error) {
    $("connection").classList.add("offline");$("connection-text").textContent="Chưa kết nối";status(error.message,true);
  }
  await Promise.allSettled([loadEvaluation(),loadExtraEvaluation(),loadRejectionEvaluation(),loadTrialSummary()]);
}

const initialTab=Object.entries(TAB_HASHES).find(([,hash])=>hash===location.hash)?.[0]||"recognize";
setTab(initialTab);
window.addEventListener("beforeunload",()=>{
  state.cameraToken++;state.sourceToken++;clearInterval(state.timer);
  if(state.recorder?.state==="recording")try{state.recorder.stop();}catch{}
  state.stream?.getTracks().forEach(track=>track.stop());if(state.objectUrl)URL.revokeObjectURL(state.objectUrl);
});
initialize();
