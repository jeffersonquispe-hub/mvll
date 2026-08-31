// ============================================================
// ELEMENTOS DEL DOM
// ============================================================
const textInput        = document.getElementById('textInput');
const sendBtn          = document.getElementById('sendBtn');
const micBtn           = document.getElementById('micBtn');
const chatMessages     = document.getElementById('chatMessages');
const avatarImageChat  = document.getElementById('avatarImageChat');
const avatarRingChat   = document.getElementById('avatarRingChat');
const asrStatusText    = document.getElementById('asrStatusText');
const startCallBtn     = document.getElementById('startCallBtn');

// Ventana "Sobre el proyecto" / Feedback
const infoBtn           = document.getElementById('infoBtn');
const infoModal         = document.getElementById('infoModal');
const infoModalCloseBtn = document.getElementById('infoModalCloseBtn');
const feedbackTextarea  = document.getElementById('feedbackTextarea');
const feedbackSubmitBtn = document.getElementById('feedbackSubmitBtn');
const feedbackStatus    = document.getElementById('feedbackStatus');

// Llamada
const chatView         = document.getElementById('chatView');
const callView         = document.getElementById('callView');
const callStatus       = document.getElementById('callStatus');
const callAvatarRing   = document.getElementById('callAvatarRing');
const callAvatarWrapper = callView.querySelector('.call-avatar-wrapper');
const hangupBtn        = document.getElementById('hangupBtn');
const muteBtn          = document.getElementById('muteBtn');

// ============================================================
// ESTADO GLOBAL
// ============================================================
let livekitRoom     = null;
let livekitUrl      = '';
let isMuted         = false;
let isRecording     = false;
let speechRecognition = null;
let callConnected   = false; // true una vez que la llamada está realmente en curso (no "conectando...")

// ============================================================
// INICIALIZACIÓN
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    checkBackendConfig();
    initSpeechRecognition();
});

// ============================================================
// LISTENERS
// ============================================================
function setupEventListeners() {
    // Chat: enviar mensaje
    sendBtn.addEventListener('click', sendMessage);
    textInput.addEventListener('keypress', e => {
        if (e.key === 'Enter') sendMessage();
    });

    // Botón de enviar reactivo: apagado mientras no haya nada que enviar
    textInput.addEventListener('input', () => {
        sendBtn.disabled = textInput.value.trim().length === 0;
    });

    // En mobile, el teclado virtual puede tapar el input recién enfocado
    textInput.addEventListener('focus', () => {
        setTimeout(() => textInput.scrollIntoView({ behavior: 'smooth', block: 'center' }), 300);
    });

    // Chat: micrófono (ASR fallback)
    micBtn.addEventListener('click', toggleRecording);

    // Chat: chips de sugerencias
    document.querySelectorAll('.chip-v').forEach(chip => {
        chip.addEventListener('click', e => {
            const prompt = e.target.getAttribute('data-prompt');
            if (livekitRoom) {
                showUserMessage(prompt);
                sendTextToLiveKitAgent(prompt);
            } else {
                textInput.value = prompt;
                sendMessage();
            }
        });
    });

    // INICIAR LLAMADA
    startCallBtn.addEventListener('click', startCall);

    // COLGAR
    hangupBtn.addEventListener('click', endCall);

    // SILENCIAR
    muteBtn.addEventListener('click', toggleMute);

    // VENTANA "SOBRE EL PROYECTO"
    infoBtn.addEventListener('click', openInfoModal);
    infoModalCloseBtn.addEventListener('click', closeInfoModal);
    infoModal.addEventListener('click', e => {
        if (e.target === infoModal) closeInfoModal();
    });
    feedbackSubmitBtn.addEventListener('click', submitFeedback);
}

// ============================================================
// VENTANA "SOBRE EL PROYECTO" / FEEDBACK
// ============================================================
function openInfoModal() {
    infoModal.classList.remove('hidden');
}

function closeInfoModal() {
    infoModal.classList.add('hidden');
}

async function submitFeedback() {
    const message = feedbackTextarea.value.trim();
    if (!message) return;

    feedbackSubmitBtn.disabled = true;
    feedbackStatus.textContent = 'Enviando...';

    try {
        const res = await fetch('/api/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });
        if (!res.ok) throw new Error(`Error ${res.status}`);

        feedbackTextarea.value = '';
        feedbackStatus.textContent = '¡Gracias por tu feedback!';
    } catch (e) {
        console.error('Error enviando feedback:', e);
        feedbackStatus.textContent = 'No se pudo enviar. Intenta de nuevo.';
    } finally {
        feedbackSubmitBtn.disabled = false;
        setTimeout(() => { feedbackStatus.textContent = ''; }, 4000);
    }
}

// ============================================================
// CONFIGURACIÓN DEL BACKEND
// ============================================================
async function checkBackendConfig() {
    try {
        const res = await fetch('/api/config');
        if (res.ok) {
            const data = await res.json();
            livekitUrl = data.livekit_url || '';
            chatMessages.innerHTML = '';
            const msg = livekitUrl
                ? 'Escríbele o inicia una llamada de voz en tiempo real.'
                : 'Escribe tu pregunta para comenzar a conversar.';
            showWelcomeMessage(msg);

            // Mostrar/ocultar botón de llamada según disponibilidad de LiveKit
            startCallBtn.style.display = livekitUrl ? 'flex' : 'none';
        }
    } catch {
        showSystemMessage('Conectando con el servidor...');
        startCallBtn.style.display = 'none';
    }
}

// ============================================================
// MODO LLAMADA — TRANSICIÓN DE VISTAS
// ============================================================
function startCall() {
    chatView.classList.add('hidden');
    callView.classList.remove('hidden');
    setCallStatus('Conectando con Don Mario...');
    connectToLiveKit();
}

function endCall() {
    disconnectFromLiveKit();
    callView.classList.add('hidden');
    chatView.classList.remove('hidden');
    setSpeaking(false);
    callConnected = false;
    isMuted = false;
    muteBtn.classList.remove('muted');
    muteBtn.querySelector('i').className = 'fa-solid fa-microphone';
}

function setCallStatus(text) {
    if (callStatus) callStatus.textContent = text;
}

// ============================================================
// MODO LIVEKIT (WebRTC)
// ============================================================
async function connectToLiveKit() {
    if (livekitRoom) await disconnectFromLiveKit();
    setCallStatus('Estableciendo canal de voz...');

    try {
        const uniqueRoom = `mvll-room-${Date.now()}-${Math.random().toString(36).substring(2, 6)}`;
        const res = await fetch(`/api/livekit/token?room=${uniqueRoom}`);
        if (!res.ok) throw new Error('No se pudo obtener el token de LiveKit');
        const { token } = await res.json();

        livekitRoom = new LivekitClient.Room({
            adaptiveStream: true,
            dynacast: true
        });

        livekitRoom.on(LivekitClient.RoomEvent.TrackSubscribed, (track, _pub, participant) => {
            if (track.kind === 'audio') {
                const el = track.attach();
                document.body.appendChild(el);
            }
        });

        livekitRoom.on(LivekitClient.RoomEvent.ActiveSpeakersChanged, speakers => {
            const localId = livekitRoom.localParticipant.identity;
            const agentSpeaking = speakers.some(s => s.identity !== localId);
            setSpeaking(agentSpeaking);
        });

        livekitRoom.on(LivekitClient.RoomEvent.Disconnected, () => {
            if (!callView.classList.contains('hidden')) {
                setCallStatus('Llamada finalizada.');
                setTimeout(endCall, 1200);
            }
        });

        await livekitRoom.connect(livekitUrl, token);
        await livekitRoom.localParticipant.setMicrophoneEnabled(true);
        callConnected = true;
        setCallStatus('Te escucho — habla libremente');

    } catch (e) {
        console.error('[LiveKit] Error:', e);
        setCallStatus(micErrorMessage(e));
        setTimeout(endCall, 5000);
    }
}

// Traduce los errores de captura de micrófono a algo entendible para el usuario
function micErrorMessage(e) {
    const name = e && e.name;
    switch (name) {
        case 'NotReadableError':
        case 'AbortError':
            return 'No se pudo acceder al micrófono. Cierra cualquier app que lo esté usando (Zoom, Discord, OBS, Teams…) y vuelve a intentarlo.';
        case 'NotAllowedError':
        case 'SecurityError':
            return 'Permiso de micrófono denegado. Actívalo en el candado de la barra de direcciones y recarga la página.';
        case 'NotFoundError':
        case 'OverconstrainedError':
            return 'No se detectó ningún micrófono. Conecta uno y comprueba el dispositivo de entrada en la configuración de sonido.';
        default:
            return `No se pudo iniciar la llamada de voz${name ? ` (${name})` : ''}.`;
    }
}

async function disconnectFromLiveKit() {
    if (livekitRoom) {
        try { await livekitRoom.disconnect(); } catch {}
        livekitRoom = null;
    }
}

function sendTextToLiveKitAgent(text) {
    if (!livekitRoom) return;
    try {
        const data = new TextEncoder().encode(JSON.stringify({ text }));
        livekitRoom.localParticipant.publishData(data, { reliable: true });
    } catch (e) {
        console.warn('[LiveKit] Error enviando datos:', e);
    }
}

// ============================================================
// ANIMACIÓN DE HABLA (compartida entre vistas)
// ============================================================
function setSpeaking(speaking) {
    if (avatarRingChat) avatarRingChat.classList.toggle('speaking', speaking);
    if (callAvatarRing) callAvatarRing.classList.toggle('speaking', speaking);
    if (callAvatarWrapper) callAvatarWrapper.classList.toggle('speaking', speaking);

    // No pisar mensajes de "Conectando..." mientras se establece la llamada
    if (callConnected) {
        setCallStatus(speaking ? 'Mario está hablando...' : 'Te escucho — habla libremente');
    }
}

// ============================================================
// SILENCIAR MICRÓFONO EN LLAMADA
// ============================================================
async function toggleMute() {
    if (!livekitRoom) return;
    isMuted = !isMuted;
    await livekitRoom.localParticipant.setMicrophoneEnabled(!isMuted);
    muteBtn.classList.toggle('muted', isMuted);
    muteBtn.querySelector('i').className = isMuted
        ? 'fa-solid fa-microphone-slash'
        : 'fa-solid fa-microphone';
}

// ============================================================
// MODO HTTP — CHAT DE TEXTO (SOLO TEXTO, SIN AUDIO)
// ============================================================
async function sendMessage() {
    const query = textInput.value.trim();
    if (!query) return;

    textInput.value = '';
    sendBtn.disabled = true;
    showUserMessage(query);

    const botEl = showBotLoadingMessage();
    const bubble = botEl.querySelector('.message-bubble');

    let cursorEl = null;

    function initStreamBubble() {
        bubble.innerHTML = '';
        cursorEl = document.createElement('span');
        cursorEl.className = 'stream-cursor';
        cursorEl.textContent = '▍';
        bubble.appendChild(cursorEl);
    }

    async function typeToken(text) {
        for (const char of text) {
            bubble.insertBefore(document.createTextNode(char), cursorEl);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            await new Promise(r => setTimeout(r, 14));
        }
    }

    function finishStream() {
        if (cursorEl) cursorEl.remove();
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    try {
        const response = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: query })
        });

        if (!response.ok) throw new Error(`Error ${response.status}`);

        initStreamBubble();

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            let currentEvent = '';
            for (const line of lines) {
                if (line.startsWith('event: ')) {
                    currentEvent = line.slice(7).trim();
                } else if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));

                        if (currentEvent === 'status') {
                            const ld = bubble.querySelector('.loading-dots');
                            if (ld) ld.textContent = 'Don Mario está ' + data.text;

                        } else if (currentEvent === 'token') {
                            if (bubble.querySelector('.loading-dots')) initStreamBubble();
                            await typeToken(data.text);

                        } else if (currentEvent === 'done') {
                            finishStream();

                        } else if (currentEvent === 'error') {
                            finishStream();
                            bubble.textContent = 'Disculpa, tuve una dificultad al formular mi pensamiento.';
                        }
                    } catch {}
                }
            }
        }

        finishStream();
        addCopyButton(botEl, bubble);

    } catch (e) {
        console.error('Error en streaming:', e);
        bubble.textContent = 'Disculpa, hubo un error al conectar con el servidor.';
    } finally {
        sendBtn.disabled = textInput.value.trim().length === 0;
    }
}

// ============================================================
// ASR LOCAL (webkitSpeechRecognition)
// ============================================================
function initSpeechRecognition() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) return;
    speechRecognition = new SR();
    speechRecognition.continuous = false;
    speechRecognition.lang = 'es-ES';
    speechRecognition.interimResults = false;

    speechRecognition.onstart = () => {
        showAsrStatus('Escuchando...');
        micBtn.classList.add('recording');
    };

    speechRecognition.onresult = e => {
        textInput.value = e.results[0][0].transcript;
        hideAsrStatus();
        sendMessage();
    };

    speechRecognition.onerror = () => {
        hideAsrStatus();
        micBtn.classList.remove('recording');
    };

    speechRecognition.onend = () => {
        micBtn.classList.remove('recording');
        hideAsrStatus();
        isRecording = false;
    };
}

function toggleRecording() {
    if (isRecording) {
        if (speechRecognition) speechRecognition.stop();
        isRecording = false;
    } else {
        if (speechRecognition) {
            isRecording = true;
            speechRecognition.start();
        } else {
            showSystemMessage('El reconocimiento de voz no está soportado en este navegador.');
        }
    }
}

// ============================================================
// HELPERS DE UI — MENSAJES
// ============================================================
function showUserMessage(text) {
    const el = document.createElement('div');
    el.className = 'message user-message';
    el.innerHTML = `<div class="message-bubble">${escapeHtml(text)}</div>`;
    chatMessages.appendChild(el);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showWelcomeMessage(subtext) {
    const el = document.createElement('div');
    el.className = 'welcome-card';
    el.innerHTML = `
        <i class="fa-solid fa-feather-pointed"></i>
        <h3>Biblioteca de Don Mario</h3>
        <p>${escapeHtml(subtext)}</p>
    `;
    chatMessages.appendChild(el);
}

function showSystemMessage(text) {
    const el = document.createElement('div');
    el.className = 'message system-msg';
    el.innerHTML = `<span>${escapeHtml(text)}</span>`;
    chatMessages.appendChild(el);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showBotLoadingMessage() {
    const el = document.createElement('div');
    el.className = 'message assistant-message';
    el.innerHTML = `<div class="message-bubble"><span class="loading-dots">Don Mario está reflexionando<span>.</span><span>.</span><span>.</span></span></div>`;
    chatMessages.appendChild(el);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return el;
}

function addCopyButton(container, bubble) {
    if (!navigator.clipboard) return; // requiere contexto seguro (https/localhost)

    const btn = document.createElement('button');
    btn.className = 'copy-msg-btn';
    btn.type = 'button';
    btn.title = 'Copiar esta respuesta';
    btn.setAttribute('aria-label', 'Copiar esta respuesta');
    btn.innerHTML = '<i class="fa-regular fa-copy"></i><span>Copiar</span>';

    btn.addEventListener('click', async () => {
        try {
            await navigator.clipboard.writeText(bubble.textContent.trim());
            btn.classList.add('copied');
            btn.innerHTML = '<i class="fa-solid fa-check"></i><span>Copiado</span>';
            setTimeout(() => {
                btn.classList.remove('copied');
                btn.innerHTML = '<i class="fa-regular fa-copy"></i><span>Copiar</span>';
            }, 1600);
        } catch (e) {
            console.warn('No se pudo copiar:', e);
        }
    });

    container.appendChild(btn);
}

function showAsrStatus(text) {
    asrStatusText.textContent = text;
    asrStatusText.classList.add('visible');
}

function hideAsrStatus() {
    asrStatusText.textContent = '';
    asrStatusText.classList.remove('visible');
}

function escapeHtml(str) {
    return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
