  // ── DOM refs ────────────────────────────────────────────────
  const dot              = document.getElementById("dot");
  const statusText       = document.getElementById("status-text");
  const stateNoPhoto     = document.getElementById("state-no-photo");
  const stateHasPhoto    = stateNoPhoto;
  const stateAnalyzing   = document.getElementById("state-analyzing");
  const audioPanel       = document.getElementById("audio-panel");
  const debugPanel       = document.getElementById("debug-panel");
  const introGate        = document.getElementById("intro-gate");
  const permissionGate   = document.getElementById("permission-gate");
  const introCopy        = document.getElementById("intro-copy");
  const introNote        = document.getElementById("intro-note");
  const btnIntroContinue = document.getElementById("btn-intro-continue");
  const btnEnableAudio   = document.getElementById("btn-enable-audio");
  const btnDebug         = document.getElementById("btn-debug");
  const btnDebugClose    = document.getElementById("btn-debug-close");
  const debugBackdrop    = document.getElementById("debug-backdrop");
  const cameraPanel      = document.getElementById("camera-panel");
  const cameraPreview    = document.getElementById("camera-preview");
  const cameraStatusText = document.getElementById("camera-status-text");
  const btnCameraCaptureSurface = document.getElementById("btn-camera-capture-surface");
  const ttsAudio         = document.getElementById("tts-audio");
  const debugActiveImage = document.getElementById("debug-active-image");
  const debugDepthImage  = document.getElementById("debug-depth-image");
  const debugNavigatorImage = document.getElementById("debug-navigator-image");
  const debugMeasurements = document.getElementById("debug-measurements");
  const debugHistory     = document.getElementById("debug-history");

  // ── State ────────────────────────────────────────────────────
  let depthOpen        = false;
  let hasPhoto         = false;
  let isRecording      = false;
  let mediaRecorder    = null;
  let audioChunks      = [];
  let queryWithImage   = true;
  let currentImageFile = null;
  let currentCameraStream = null;
  let isCapturingPhoto = false;
  let _prevState       = null;   // which state div was showing before analyzing
  let _welcomePlayed   = false;
  let _introExplained  = false;
  let _isGrantingPermissions = false;
  let _photoHintPlayed = false;
  let _askHintPlayed   = false;
  let _announcementTimerIds = [];
  let _playbackWatchdogId = null;
  let _debugState      = null;
  let sessionId = localStorage.getItem("ss_sid") || crypto.randomUUID();
  localStorage.setItem("ss_sid", sessionId);
  const WELCOME_TEXT = "Welcome to Gemma Guide. First press the button in the center of the screen to allow microphone and camera access, then click Allow on the popup. After that, Take Photo is the top part of the screen and Ask Question is the bottom part of the screen. You can use them in any order. After the response is played, tap once to go back to the main page.";
  const INTRO_GUIDE_TEXT = "Welcome to Gemma Guide. Please listen to the full instructions before pressing again. First press the button in the center of the screen to allow microphone and camera access, then click Allow on the popup. After that, Take Photo is the top part of the screen and Ask Question is the bottom part of the screen. You can use them in any order. After the response is played, tap once to go back to the main page. You can now continue with the access button.";
  const PHOTO_HINT_TEXT = "Take Photo. Press the button to open the camera, point the phone, then press anywhere on the screen to take the photo.";
  const ASK_HINT_TEXT = "Ask Question. Press once, speak, then press again to stop recording and send.";

  const cueAudio = {
    dictationStart: new Audio("/designs/audio/misc_audio/mic_start_trimmed_cut.mp3"),
    dictationStop:  new Audio("/designs/audio/dictation_stop.mp3"),
    woosh:          new Audio("/designs/audio/woosh.mp3"),
    loading:        new Audio("/designs/audio/misc_audio/model_load.mp3"),
    imageStart:     new Audio("/designs/audio/misc_audio/image_start_trimmed.mp3"),
  };
  cueAudio.loading.loop = true;
  cueAudio.loading.volume = 0.35;
  cueAudio.woosh.volume = 0.9;
  cueAudio.dictationStart.volume = 0.9;
  cueAudio.dictationStop.volume = 0.9;
  cueAudio.imageStart.volume = 0.9;

  // ── iOS audio unlock ─────────────────────────────────────────
  // Play a silent sound on first user touch so subsequent .play() calls work.
  let _audioUnlocked = false;
  async function _unlockAudio() {
    if (_audioUnlocked) return;
    _audioUnlocked = true;
    const silent = "data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA";
    ttsAudio.src = silent;
    try {
      await ttsAudio.play();
      ttsAudio.pause();
      ttsAudio.src = "";
    } catch (_) {}
    ttsAudio.src = "";
  }

  // ── Helpers ──────────────────────────────────────────────────
  function explainIntroAndRevealMic() {
    if (_introExplained) return;
    _introExplained = true;
    btnIntroContinue.disabled = true;
    scheduleAnnouncement(() => speakAnnouncement(INTRO_GUIDE_TEXT), 60);
    scheduleAnnouncement(() => {
      introGate.classList.add("hidden");
      permissionGate.classList.remove("hidden");
      btnEnableAudio.focus();
    }, 120);
  }

  function speakAnnouncement(text, { interrupt = true } = {}) {
    if (!("speechSynthesis" in window)) return;
    console.log("[audio] speakAnnouncement", { text, interrupt });
    if (interrupt) window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;
    utterance.onstart = () => console.log("[audio] speech start", { text });
    utterance.onend = () => console.log("[audio] speech end", { text });
    utterance.onerror = (event) => console.warn("[audio] speech error", { text, event });
    window.speechSynthesis.speak(utterance);
  }

  function speakAnnouncementAsync(text, { interrupt = true } = {}) {
    if (!("speechSynthesis" in window)) return Promise.resolve();
    console.log("[audio] speakAnnouncementAsync", { text, interrupt });
    if (interrupt) window.speechSynthesis.cancel();
    return new Promise((resolve) => {
      const utterance = new SpeechSynthesisUtterance(text);
      let settled = false;
      const finish = () => {
        if (settled) return;
        settled = true;
        resolve();
      };
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      utterance.volume = 1.0;
      utterance.onstart = () => console.log("[audio] speech start", { text });
      utterance.onend = () => {
        console.log("[audio] speech end", { text });
        finish();
      };
      utterance.onerror = (event) => {
        console.warn("[audio] speech error", { text, event });
        finish();
      };
      window.speechSynthesis.speak(utterance);
    });
  }

  function playCue(name) {
    const audio = cueAudio[name];
    if (!audio) return;
    try {
      audio.pause();
      audio.currentTime = 0;
      audio.play().catch(() => {});
    } catch (_) {}
  }

  function hapticTap(pattern = 18) {
    try {
      if (navigator.vibrate) navigator.vibrate(pattern);
    } catch (_) {}
  }

  function scheduleAnnouncement(callback, delay = 0) {
    const id = window.setTimeout(() => {
      _announcementTimerIds = _announcementTimerIds.filter((timerId) => timerId !== id);
      callback();
    }, delay);
    _announcementTimerIds.push(id);
    return id;
  }

  function clearPendingAnnouncementTimers() {
    _announcementTimerIds.forEach((id) => window.clearTimeout(id));
    _announcementTimerIds = [];
  }

  function stopAnnouncements() {
    clearPendingAnnouncementTimers();
    try {
      window.speechSynthesis?.cancel();
    } catch (_) {}
  }

  function clearPlaybackWatchdog() {
    if (_playbackWatchdogId !== null) {
      window.clearTimeout(_playbackWatchdogId);
      _playbackWatchdogId = null;
    }
  }

  function startLoadingCue() {
    try {
      cueAudio.loading.currentTime = 0;
      cueAudio.loading.play().catch(() => {});
    } catch (_) {}
  }

  function stopLoadingCue() {
    try {
      cueAudio.loading.pause();
      cueAudio.loading.currentTime = 0;
    } catch (_) {}
  }

  function resetConversationState() {
    sessionId = crypto.randomUUID();
    localStorage.setItem("ss_sid", sessionId);
    currentImageFile = null;
    isCapturingPhoto = false;
    hasPhoto = false;
    stateNoPhoto.style.display = "flex";
    resetAskButtons();
    setStatus("ready");
    stopLoadingCue();
    stopCameraPreview();
    stopAnnouncements();
    clearPlaybackWatchdog();
    ttsAudio.pause();
    ttsAudio.src = "";
    _debugState = null;
  }

  function imageSrcFromB64(b64) {
    return b64 ? `data:image/jpeg;base64,${b64}` : "";
  }

  function setDebugImage(node, b64) {
    if (!b64) {
      node.style.display = "none";
      node.removeAttribute("src");
      return;
    }
    node.src = imageSrcFromB64(b64);
    node.style.display = "block";
  }

  function renderDebugPanel(debug) {
    _debugState = debug || null;
    setDebugImage(debugActiveImage, debug?.active_image_b64 || null);
    setDebugImage(debugDepthImage, debug?.depth_b64 || null);
    setDebugImage(debugNavigatorImage, debug?.navigator_image_b64 || null);

    const measurements = Array.isArray(debug?.measurements) ? debug.measurements : [];
    debugMeasurements.textContent = measurements.length
      ? measurements.map((m, idx) => `${idx + 1}. ${m.requested_class_name || m.class_name} — ${m.tips_distance_m} m, ${m.direction}`).join("\n")
      : "No measurements yet.";

    debugHistory.innerHTML = "";
    const historyItems = Array.isArray(debug?.history) ? debug.history : [];
    if (!historyItems.length) {
      const empty = document.createElement("div");
      empty.className = "debug-text";
      empty.textContent = "No conversation history yet.";
      debugHistory.appendChild(empty);
      return;
    }
    historyItems.forEach((item) => {
      const turn = document.createElement("div");
      turn.className = "debug-turn";
      const role = document.createElement("div");
      role.className = "debug-role";
      role.textContent = item.role || "unknown";
      const text = document.createElement("div");
      text.className = "debug-text";
      text.textContent = item.text || "";
      turn.appendChild(role);
      turn.appendChild(text);
      debugHistory.appendChild(turn);
    });
  }

  function openDebugPanel() {
    renderDebugPanel(_debugState);
    debugPanel.classList.add("open");
  }

  function closeDebugPanel() {
    debugPanel.classList.remove("open");
  }

  function dismissIntroGate() {
    introGate.classList.add("hidden");
  }

  function dismissPermissionGate() {
    permissionGate.classList.add("hidden");
  }

  async function enableAudioAndEnter() {
    if (_isGrantingPermissions) return;
    _isGrantingPermissions = true;
    btnEnableAudio.disabled = true;
    setStatus("enabling audio…", true);
    try {
      const unlockPromise = _unlockAudio();
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
        video: {
          facingMode: { ideal: "environment" },
        },
      });
      dismissPermissionGate();
      await unlockPromise;
      stream.getTracks().forEach(track => track.stop());
      resetConversationState();
      setStatus("audio ready", true);
      _welcomePlayed = true;
    } catch (err) {
      console.warn("Microphone or camera access not granted:", err);
      setStatus("permissions denied", false);
    } finally {
      _isGrantingPermissions = false;
      btnEnableAudio.disabled = false;
    }
  }

  function setStatus(label, live = false) {
    statusText.textContent = label;
    dot.classList.toggle("live", live);
  }

  function transitionToHasPhoto() {
    hasPhoto = true;
  }

  function showAnalyzing() {
    _prevState = hasPhoto ? stateHasPhoto : stateNoPhoto;
    _prevState.style.display = "none";
    stateAnalyzing.style.display = "flex";
    setStatus("analyzing…", true);
  }

  function hideAnalyzing() {
    stateAnalyzing.style.display = "none";
    if (_prevState) { _prevState.style.display = "flex"; _prevState = null; }
    setStatus("ready");
  }

  function b64ToBlob(b64, mime) {
    const bin = atob(b64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    return new Blob([arr], { type: mime });
  }

  function resetAskButtons() {
    const a = document.getElementById("btn-ask");
    if (a) { a.querySelector(".btn-label").textContent = "Ask Question"; a.classList.remove("recording"); }
  }

  // ── Camera ───────────────────────────────────────────────────
  async function startCameraPreview() {
    if (currentCameraStream) return;
    const constraints = {
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1920 },
        height: { ideal: 1080 },
      },
      audio: false,
    };
    currentCameraStream = await navigator.mediaDevices.getUserMedia(constraints);
    cameraPreview.srcObject = currentCameraStream;
    await cameraPreview.play();
  }

  function stopCameraPreview() {
    if (currentCameraStream) {
      currentCameraStream.getTracks().forEach(track => track.stop());
      currentCameraStream = null;
    }
    cameraPreview.pause();
    cameraPreview.srcObject = null;
    cameraPanel.classList.remove("open");
  }

  async function triggerCamera() {
    if (isCapturingPhoto) return;
    if (!_photoHintPlayed) {
      _photoHintPlayed = true;
      speakAnnouncement(PHOTO_HINT_TEXT);
      return;
    }
    playCue("imageStart");
    cameraStatusText.textContent = "Opening camera preview…";
    cameraPanel.classList.add("open");
    try {
      await startCameraPreview();
      cameraStatusText.textContent = "Camera preview ready. Tap anywhere on the screen to take the photo.";
      btnCameraCaptureSurface.focus();
    } catch (err) {
      console.error("Camera preview failed:", err);
      cameraStatusText.textContent = "Could not open the camera preview.";
      setStatus("camera unavailable", false);
      stopCameraPreview();
    }
  }

  function canvasToJpegBlob(canvas, quality = 0.92) {
    return new Promise((resolve) => {
      canvas.toBlob((blob) => {
        if (blob) {
          resolve(blob);
          return;
        }
        try {
          const dataUrl = canvas.toDataURL("image/jpeg", quality);
          const base64 = dataUrl.split(",")[1] || "";
          const binary = atob(base64);
          const bytes = new Uint8Array(binary.length);
          for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
          resolve(new Blob([bytes], { type: "image/jpeg" }));
        } catch (_) {
          resolve(null);
        }
      }, "image/jpeg", quality);
    });
  }

  async function captureCameraFrame() {
    if (isCapturingPhoto) return;
    if (!cameraPreview.videoWidth || !cameraPreview.videoHeight) {
      speakAnnouncement("Camera is not ready yet. Please try again.");
      return;
    }
    isCapturingPhoto = true;
    btnCameraCaptureSurface.disabled = true;
    cameraStatusText.textContent = "Capturing photo…";
    const canvas = document.createElement("canvas");
    canvas.width = cameraPreview.videoWidth;
    canvas.height = cameraPreview.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      isCapturingPhoto = false;
      btnCameraCaptureSurface.disabled = false;
      speakAnnouncement("I could not capture the photo. Please try again.");
      return;
    }
    ctx.drawImage(cameraPreview, 0, 0, canvas.width, canvas.height);
    const blob = await canvasToJpegBlob(canvas, 0.92);
    if (!blob || !blob.size) {
      isCapturingPhoto = false;
      btnCameraCaptureSurface.disabled = false;
      cameraStatusText.textContent = "Photo capture failed.";
      speakAnnouncement("I could not capture the photo. Please try again.");
      return;
    }
    currentImageFile = new File([blob], `capture-${Date.now()}.jpg`, { type: "image/jpeg" });
    console.log("[camera] captured image", {
      name: currentImageFile.name,
      size: currentImageFile.size,
      type: currentImageFile.type,
      width: canvas.width,
      height: canvas.height,
    });
    sessionId = crypto.randomUUID();
    localStorage.setItem("ss_sid", sessionId);
    stopCameraPreview();
    isCapturingPhoto = false;
    btnCameraCaptureSurface.disabled = false;
    setStatus("photo taken", true);
    speakAnnouncement("Photo taken. You can now ask a question.", { interrupt: false });
    transitionToHasPhoto();
  }

  // ── Recording ────────────────────────────────────────────────
  async function startRecording(withImage) {
    queryWithImage = withImage;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunks = [];
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "";
      mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});
      mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
      mediaRecorder.onstop = () => {
        stream.getTracks().forEach(t => t.stop());
        submitQuery(queryWithImage);
      };
      mediaRecorder.start();
      isRecording = true;
      playCue("dictationStart");
    } catch (err) {
      setStatus("mic denied", false);
      resetAskButtons();
      console.error(err);
    }
  }

  function stopRecording() {
    if (mediaRecorder && isRecording) {
      playCue("dictationStop");
      mediaRecorder.stop();
      isRecording = false;
    }
  }

  function toggleRecording(btn, withImage) {
    if (withImage && isCapturingPhoto) {
      speakAnnouncement("Please wait. I am still capturing the photo.");
      return;
    }
    if (!isRecording) {
      if (!_askHintPlayed) {
        _askHintPlayed = true;
        speakAnnouncement(ASK_HINT_TEXT);
        return;
      }
      btn.querySelector(".btn-label").textContent = "Stop & Send";
      btn.classList.add("recording");
      setStatus("recording…", true);
      startRecording(withImage);
    } else {
      stopRecording();
      resetAskButtons();
    }
  }

  // ── Submit ───────────────────────────────────────────────────
  async function submitQuery(withImage) {
    if (withImage && isCapturingPhoto) {
      speakAnnouncement("Please wait. I am still capturing the photo.");
      return;
    }
    const attachImage = withImage && !!currentImageFile;
    if (withImage && !attachImage) {
      console.log("[query] no current image available; submitting audio/text only", { sessionId });
    }
    showAnalyzing();
    stopAnnouncements();
    const sendText = audioChunks.length > 0
      ? (attachImage ? "Sending audio with photo." : "Sending audio only.")
      : (attachImage ? "Sending text with photo." : "Sending text only.");
    await speakAnnouncementAsync(sendText);
    playCue("woosh");
    startLoadingCue();

    const fd = new FormData();
    fd.append("session_id", sessionId);

    if (attachImage && currentImageFile) {
      console.log("[query] attaching image", {
        sessionId,
        name: currentImageFile.name,
        size: currentImageFile.size,
        type: currentImageFile.type,
      });
      fd.append("image", currentImageFile, currentImageFile.name);
    }

    if (audioChunks.length > 0) {
      const mime = mediaRecorder?.mimeType || "audio/webm";
      const ext  = mime.includes("mp4") ? "mp4" : "webm";
      fd.append("audio", new Blob(audioChunks, { type: mime }), `q.${ext}`);
    }

    audioChunks = [];

    try {
      const resp = await fetch("/api/query", { method: "POST", body: fd });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      console.log("[query] response", {
        route: data.route,
        responseLength: (data.response || "").length,
        hasDepth: Boolean(data.depth_b64),
      });
      renderDebugPanel(data.debug || null);

      if (data.metrics) console.log("Gemma Guide metrics", data.metrics);

      if (data.route === "restart") {
        resetConversationState();
        speakAnnouncement(data.response || "Starting a new scene.");
        hideAnalyzing();
        return;
      }

      stopLoadingCue();
      hideAnalyzing();
      openAudioPanel(false);
      if (data.response) {
        await speakAnnouncementAsync(data.response);
      } else {
        speakAnnouncement("I did not receive a response to play.");
      }
      setStatus("ready");
    } catch (err) {
      console.error("Query failed:", err);
      stopAnnouncements();
      stopLoadingCue();
      hideAnalyzing();
      setStatus("error", false);
      speakAnnouncement("Could not reach the server. Please check your connection.");
    }
  }

  // ── Button wiring ────────────────────────────────────────────
  document.getElementById("btn-photo-first").addEventListener("click", () => {
    hapticTap();
    triggerCamera();
  });
  document.getElementById("btn-ask").addEventListener("click", function () {
    hapticTap();
    toggleRecording(this, true);
  });
  btnCameraCaptureSurface.addEventListener("click", () => {
    hapticTap([14, 18, 14]);
    captureCameraFrame();
  });

  // ── Audio panel ──────────────────────────────────────────────
  function openAudioPanel(playMedia = true) {
    audioPanel.classList.add("open");
    setStatus("playing…", true);
    stopLoadingCue();
    stopAnnouncements();
    if (playMedia) {
      // Try immediate play; if audio hasn't loaded yet fall back to canplay event.
      ttsAudio.play().catch(() => {
        ttsAudio.addEventListener("canplay", () => {
          ttsAudio.play().catch(e => console.warn("Autoplay blocked:", e));
        }, { once: true });
      });
    }
    document.getElementById("btn-back-main").focus();
  }

  function closeAudioPanel() {
    audioPanel.classList.remove("open");
    ttsAudio.pause();
    stopAnnouncements();
    setStatus("ready");
  }

  document.getElementById("btn-back-main").addEventListener("click", () => {
    hapticTap();
    closeAudioPanel();
  });
  document.getElementById("panel-backdrop").addEventListener("click", () => {
    hapticTap();
    closeAudioPanel();
  });

  ttsAudio.addEventListener("ended", () => setStatus("ready"));

  btnDebug.addEventListener("click", () => {
    hapticTap();
    openDebugPanel();
  });
  btnDebugClose.addEventListener("click", () => {
    hapticTap();
    closeDebugPanel();
  });
  debugBackdrop.addEventListener("click", closeDebugPanel);

  btnIntroContinue.addEventListener("click", () => {
    hapticTap();
    explainIntroAndRevealMic();
  });
  btnEnableAudio.addEventListener("click", () => {
    hapticTap();
    enableAudioAndEnter();
  });
