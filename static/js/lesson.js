// Lesson.js - Interactive Lesson with Pause/Play and Auto-scroll

/*let topics = [];
let currentTopicIndex = 0;
let accumulatedContext = '';
let isRecording = false;
let isSpeaking = false;
let isPaused = false;
let currentUtterance = null;
let currentSentenceIndex = 0;
let sentences = [];

// DOM Elements
const menuToggle = document.getElementById('menu-toggle');
const menuOverlay = document.getElementById('menu-overlay');
const hamburgerMenu = document.getElementById('hamburger-menu');
const closeMenu = document.getElementById('close-menu');

const pausePlayBtn = document.getElementById('pause-play-btn');
const controlText = document.getElementById('control-text');

const topicNumber = document.getElementById('topic-number');
const topicTitle = document.getElementById('topic-title');
const explanationText = document.getElementById('explanation-text');
const keyPoints = document.getElementById('key-points');
const exampleBox = document.getElementById('example-box');
const exampleText = document.getElementById('example-text');
const chalkboardSection = document.getElementById('chalkboard-section');

const teacherVideo = document.getElementById('teacher-video');
const understandingCheck = document.getElementById('understanding-check');
const btnYes = document.getElementById('btn-yes');
const btnNo = document.getElementById('btn-no');

const doubtSection = document.getElementById('doubt-section');
const doubtText = document.getElementById('doubt-text');
const submitDoubt = document.getElementById('submit-doubt');
const answerBox = document.getElementById('answer-box');
const answerText = document.getElementById('answer-text');
const continueBtn = document.getElementById('continue-learning');
const answerLoading = document.getElementById('answer-loading');

const methodTabs = document.querySelectorAll('.method-tab');
const typeInput = document.querySelector('.type-input');
const speakInput = document.querySelector('.speak-input');
const voiceRecordBtn = document.getElementById('voice-record-btn');
const recordStatus = document.getElementById('record-status');

// Speech Recognition
let recognition = null;
if ('webkitSpeechRecognition' in window) {
    recognition = new webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-IN';
}

// Menu Toggle
menuToggle.addEventListener('click', () => {
    hamburgerMenu.classList.add('active');
    menuOverlay.classList.add('active');
});

closeMenu.addEventListener('click', () => {
    hamburgerMenu.classList.remove('active');
    menuOverlay.classList.remove('active');
});

menuOverlay.addEventListener('click', () => {
    hamburgerMenu.classList.remove('active');
    menuOverlay.classList.remove('active');
});

// Pause/Play Control
pausePlayBtn.addEventListener('click', () => {
    if (isPaused) {
        // Resume
        isPaused = false;
        pausePlayBtn.innerHTML = '<i class="fas fa-pause"></i><span id="control-text">Pause</span>';
        
        if (isSpeaking && currentUtterance) {
            window.speechSynthesis.resume();
            teacherVideo.play();
        }
    } else {
        // Pause
        isPaused = true;
        pausePlayBtn.innerHTML = '<i class="fas fa-play"></i><span id="control-text">Play</span>';
        
        if (isSpeaking) {
            window.speechSynthesis.pause();
            teacherVideo.pause();
        }
    }
});

// Input Method Tabs
methodTabs.forEach(tab => {
    tab.addEventListener('click', () => {
        methodTabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        
        const method = tab.dataset.method;
        if (method === 'type') {
            typeInput.classList.remove('hidden');
            speakInput.classList.add('hidden');
        } else {
            speakInput.classList.remove('hidden');
            typeInput.classList.add('hidden');
        }
    });
});

// Voice Recording
voiceRecordBtn.addEventListener('click', () => {
    if (!recognition) {
        alert('Speech recognition not supported');
        return;
    }
    
    if (!isRecording) {
        recognition.start();
        isRecording = true;
        voiceRecordBtn.classList.add('recording');
        recordStatus.textContent = 'Listening... Speak now';
        
        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            doubtText.value = transcript;
        };
        
        recognition.onerror = () => {
            isRecording = false;
            voiceRecordBtn.classList.remove('recording');
            recordStatus.textContent = 'Click to start recording';
        };
        
        recognition.onend = () => {
            isRecording = false;
            voiceRecordBtn.classList.remove('recording');
            recordStatus.textContent = 'Click to start recording';
        };
    } else {
        recognition.stop();
    }
});

// Split text into sentences for synchronized scrolling
function splitIntoSentences(text) {
    return text.match(/[^\.!\?]+[\.!\?]+/g) || [text];
}

// Auto-scroll as voice reads
function scrollToElement(element) {
    if (element) {
        element.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'center',
            inline: 'nearest'
        });
    }
}

// Text-to-Speech with Auto-scroll
function speakText(text, elements, onComplete) {
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        
        const utterance = new SpeechSynthesisUtterance(text);
        const voices = window.speechSynthesis.getVoices();
        
        const indianVoice = voices.find(v => 
            v.lang.includes('en-IN') && v.name.toLowerCase().includes('female')
        ) || voices.find(v => v.lang.includes('en-IN'));
        
        if (indianVoice) utterance.voice = indianVoice;
        
        utterance.rate = 0.9;
        utterance.pitch = 1.1;
        utterance.volume = 1;
        
        // Track current element being read
        let currentElementIndex = 0;
        const sentences = splitIntoSentences(text);
        const wordsPerElement = Math.ceil(text.split(' ').length / elements.length);
        
        utterance.onboundary = (event) => {
            if (event.name === 'word' && elements.length > 0) {
                const wordIndex = Math.floor(event.charIndex / (text.length / sentences.length));
                const elementIndex = Math.min(
                    Math.floor(wordIndex / (sentences.length / elements.length)),
                    elements.length - 1
                );
                
                if (elementIndex !== currentElementIndex && elementIndex >= 0) {
                    currentElementIndex = elementIndex;
                    scrollToElement(elements[elementIndex]);
                }
            }
        };
        
        utterance.onstart = () => {
            isSpeaking = true;
            isPaused = false;
            teacherVideo.loop = true;
            teacherVideo.play();
            
            // Scroll to first element
            if (elements.length > 0) {
                scrollToElement(elements[0]);
            }
        };
        
        utterance.onend = () => {
            isSpeaking = false;
            teacherVideo.loop = false;
            teacherVideo.pause();
            
            if (onComplete) {
                onComplete();
            }
        };
        
        currentUtterance = utterance;
        window.speechSynthesis.speak(utterance);
    } else if (onComplete) {
        onComplete();
    }
}

window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();

// Load Topics
async function loadTopics() {
    try {
        const response = await fetch('/api/get-topics');
        const data = await response.json();
        
        if (data.status === 'success') {
            topics = data.topics;
            displayTopic(0);
        } else {
            alert('Error loading lesson content');
        }
    } catch (error) {
        alert('Error loading lesson content');
    }
}

// Display Topic
function displayTopic(index) {
    if (index >= topics.length) {
        window.location.href = '/quiz';
        return;
    }
    
    currentTopicIndex = index;
    const topic = topics[index];
    
    topicNumber.textContent = `Topic ${index + 1} of ${topics.length}`;
    topicTitle.textContent = `Topic ${index + 1}`;
    
    // Build content to speak (without "Topic X")
    let contentToSpeak = '';
    let elementsToScroll = [];
    
    // Explanation
    if (topic.clean_explanation) {
        const explainPara = document.createElement('p');
        explainPara.style.fontSize = '1.25rem';
        explainPara.style.lineHeight = '2';
        explainPara.textContent = topic.clean_explanation;
        explanationText.innerHTML = '';
        explanationText.appendChild(explainPara);
        
        contentToSpeak += topic.clean_explanation + '. ';
        elementsToScroll.push(explainPara);
    }
    
    accumulatedContext += '\n\n' + topic.clean_explanation;
    
    // Key points
    keyPoints.innerHTML = '';
    if (topic.key_points && topic.key_points.length > 0) {
        topic.key_points.forEach((point, idx) => {
            const li = document.createElement('li');
            li.textContent = point;
            keyPoints.appendChild(li);
            contentToSpeak += `${point}. `;
            elementsToScroll.push(li);
        });
    }
    
    // Example
    if (topic.example && topic.example.trim() !== '') {
        exampleBox.style.display = 'block';
        exampleText.textContent = topic.example;
        contentToSpeak += `For example, ${topic.example}`;
        elementsToScroll.push(exampleBox);
    } else {
        exampleBox.style.display = 'none';
    }
    
    // Reset UI
    understandingCheck.classList.add('hidden');
    doubtSection.classList.add('hidden');
    answerBox.classList.add('hidden');
    doubtText.value = '';
    isPaused = false;
    pausePlayBtn.innerHTML = '<i class="fas fa-pause"></i><span id="control-text">Pause</span>';
    
    // Scroll to top
    chalkboardSection.scrollTop = 0;
    
    // Start speaking after delay
    setTimeout(() => {
        speakText(contentToSpeak, elementsToScroll, () => {
            setTimeout(() => {
                understandingCheck.classList.remove('hidden');
                understandingCheck.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }, 500);
        });
    }, 1000);
}

btnYes.addEventListener('click', () => {
    window.speechSynthesis.cancel();
    teacherVideo.pause();
    displayTopic(currentTopicIndex + 1);
});

btnNo.addEventListener('click', () => {
    window.speechSynthesis.cancel();
    teacherVideo.pause();
    understandingCheck.classList.add('hidden');
    doubtSection.classList.remove('hidden');
    setTimeout(() => {
        doubtSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 300);
});

submitDoubt.addEventListener('click', async () => {
    const question = doubtText.value.trim();
    
    if (!question) {
        alert('Please enter your question');
        return;
    }
    
    answerLoading.classList.remove('hidden');
    
    try {
        const response = await fetch('/api/answer-doubt', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, context: accumulatedContext })
        });
        
        const data = await response.json();
        answerLoading.classList.add('hidden');
        
        if (data.status === 'success') {
            const answerPara = document.createElement('p');
            answerPara.style.lineHeight = '1.8';
            answerPara.style.fontSize = '1.125rem';
            answerPara.textContent = data.answer;
            answerText.innerHTML = '';
            answerText.appendChild(answerPara);
            answerBox.classList.remove('hidden');
            
            setTimeout(() => {
                answerBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }, 300);
            
            // Speak answer with scroll
            speakText(data.answer, [answerPara]);
        } else {
            alert('Error: ' + data.message);
        }
    } catch (error) {
        answerLoading.classList.add('hidden');
        alert('Error submitting question');
    }
});

continueBtn.addEventListener('click', () => {
    window.speechSynthesis.cancel();
    teacherVideo.pause();
    displayTopic(currentTopicIndex + 1);
});

window.addEventListener('beforeunload', () => {
    window.speechSynthesis.cancel();
    if (teacherVideo) teacherVideo.pause();
});

loadTopics();*/



// ============================================================
// Lesson.js  —  Indian Female Voice via Google Translate TTS
// Matches step_4_interactive_tutor.py flow exactly:
//   explanation → key points (revealed one-by-one) → example → Q&A
// Voice: Google Translate TTS (tl=en, tld=co.in) = Indian English female
// No audio files saved — streams directly to Audio objects.
// ============================================================

// ── State ─────────────────────────────────────────────────────
// ============================================================
// Lesson.js  —  Indian Female Voice via Microsoft Edge TTS
// Matches step_4_interactive_tutor.py flow exactly:
//   explanation → key points (revealed one-by-one) → example → Q&A
// Voice: en-IN-NeerjaNeural — Indian English female neural voice
// Audio streamed from Flask /api/tts — no files saved.
// Fallback: Web Speech API with best available Indian/female voice.
// ============================================================

// ── State ─────────────────────────────────────────────────────
let topics             = [];
let currentTopicIndex  = 0;
let accumulatedContext = '';
let isRecording        = false;
let isSpeaking         = false;
let isPaused           = false;
let currentAudio       = null;    // Active HTMLAudioElement

// ── DOM ───────────────────────────────────────────────────────
const menuToggle         = document.getElementById('menu-toggle');
const menuOverlay        = document.getElementById('menu-overlay');
const hamburgerMenu      = document.getElementById('hamburger-menu');
const closeMenuBtn       = document.getElementById('close-menu');
const pausePlayBtn       = document.getElementById('pause-play-btn');

const topicNumberEl      = document.getElementById('topic-number');
const topicTitleEl       = document.getElementById('topic-title');
const explanationText    = document.getElementById('explanation-text');
const keyPointsEl        = document.getElementById('key-points');
const exampleBox         = document.getElementById('example-box');
const exampleTextEl      = document.getElementById('example-text');
const chalkboardSection  = document.getElementById('chalkboard-section');

const teacherVideo       = document.getElementById('teacher-video');
const understandingCheck = document.getElementById('understanding-check');
const btnYes             = document.getElementById('btn-yes');
const btnNo              = document.getElementById('btn-no');

const doubtSection       = document.getElementById('doubt-section');
const doubtText          = document.getElementById('doubt-text');
const submitDoubt        = document.getElementById('submit-doubt');
const answerBox          = document.getElementById('answer-box');
const answerTextEl       = document.getElementById('answer-text');
const continueBtn        = document.getElementById('continue-learning');
const answerLoading      = document.getElementById('answer-loading');

const methodTabs         = document.querySelectorAll('.method-tab');
const typeInput          = document.querySelector('.type-input');
const speakInput         = document.querySelector('.speak-input');
const voiceRecordBtn     = document.getElementById('voice-record-btn');
const recordStatus       = document.getElementById('record-status');

// ── Speech Recognition ────────────────────────────────────────
let recognition = null;
if ('webkitSpeechRecognition' in window) {
    recognition               = new webkitSpeechRecognition();
    recognition.continuous    = false;
    recognition.interimResults = false;
    recognition.lang          = 'en-IN';
}

// ============================================================
// EDGE TTS  —  en-IN-NeerjaNeural (Indian English Female Neural)
// Flask /api/tts generates audio via edge-tts and streams MP3.
// Falls back to Web Speech API if the server request fails.
// ============================================================

/**
 * Build the URL for the Flask edge-tts endpoint.
 * Returns an MP3 of the text spoken by en-IN-NeerjaNeural.
 */
function ttsURL(text) {
    return '/api/tts?q=' + encodeURIComponent(text.trim());
}

/**
 * Web Speech API fallback — picks best Indian / female voice available.
 * Called only when the edge-tts server request fails.
 */
function speakFallback(text, onEnd) {
    if (!window.speechSynthesis) { if (onEnd) onEnd(); return; }
    window.speechSynthesis.cancel();
    const utt   = new SpeechSynthesisUtterance(text);
    const voices = window.speechSynthesis.getVoices();
    const voice  = voices.find(v => v.lang === 'en-IN' && /female/i.test(v.name))
                || voices.find(v => v.lang === 'en-IN')
                || voices.find(v => /heera|neerja/i.test(v.name))
                || voices.find(v => /female/i.test(v.name))
                || voices.find(v => v.lang.startsWith('en'))
                || null;
    if (voice) utt.voice = voice;
    utt.rate   = 0.92;
    utt.pitch  = 1.1;
    utt.volume = 1;
    if (onEnd) utt.onend = onEnd;
    window.speechSynthesis.speak(utt);
    // Warm up voice list for next call
    window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();
}

/**
 * Split text into ≤200-char chunks, breaking on sentence boundaries.
 * This mirrors how the gTTS Python library internally chunks text.
 */
function splitChunks(text) {
    const sentences = text.match(/[^.!?]+[.!?]+(?:\s|$)|[^.!?]+$/g) || [text];
    const chunks    = [];
    let   buf       = '';

    for (const raw of sentences) {
        const s = raw.trim();
        if (!s) continue;

        if (s.length > 190) {
            // Sentence itself too long — break on commas / spaces
            const parts = s.match(/.{1,185}(?:[,;\s]|$)/g) || [s];
            for (const p of parts) {
                const pt = p.trim();
                if (!pt) continue;
                if ((buf + ' ' + pt).trim().length > 190) {
                    if (buf) chunks.push(buf.trim());
                    buf = pt;
                } else {
                    buf = (buf + ' ' + pt).trim();
                }
            }
        } else if ((buf + ' ' + s).trim().length > 190) {
            if (buf) chunks.push(buf.trim());
            buf = s;
        } else {
            buf = (buf + ' ' + s).trim();
        }
    }
    if (buf) chunks.push(buf.trim());
    return chunks.filter(c => c.length > 0);
}

/**
 * Fetch a single TTS chunk from Flask edge-tts endpoint.
 * Returns a ready HTMLAudioElement, or null on failure (queue skips gracefully).
 * On failure also triggers Web Speech API fallback for that chunk.
 */
function fetchChunkAudio(text) {
    return new Promise(resolve => {
        const audio   = new Audio();
        audio.preload = 'auto';
        audio.src     = ttsURL(text);

        audio.addEventListener('canplaythrough', () => resolve(audio), { once: true });

        audio.addEventListener('error', () => {
            console.warn('[TTS] edge-tts failed, using Web Speech fallback for:', text.slice(0, 50));
            // Return a special object that plays via Web Speech when .play() is called
            resolve({
                _isFallback : true,
                _text       : text,
                currentTime : 0,
                onended     : null,
                pause       : () => { if (window.speechSynthesis) window.speechSynthesis.cancel(); },
                play        : function() {
                    return new Promise(res => {
                        speakFallback(this._text, () => {
                            if (this.onended) this.onended();
                            res();
                        });
                    });
                }
            });
        }, { once: true });

        audio.load();
    });
}

// ── Chalkboard highlight (active sentence glow) ───────────────
function setActive(el) {
    document.querySelectorAll('.tts-active').forEach(e => e.classList.remove('tts-active'));
    if (el) {
        el.classList.add('tts-active');
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}
function clearActive() {
    document.querySelectorAll('.tts-active').forEach(e => e.classList.remove('tts-active'));
}

// ── Video helpers ─────────────────────────────────────────────
function videoPlay()  { teacherVideo.loop = true;  teacherVideo.play().catch(() => {}); }
function videoPause() { teacherVideo.loop = false; teacherVideo.pause(); }

// ── Hard stop (cancel everything) ────────────────────────────
function stopAudio() {
    if (currentAudio) {
        currentAudio.onended = null;
        currentAudio.pause();
        currentAudio = null;
    }
    isSpeaking = false;
    isPaused   = false;
    clearActive();
    videoPause();
    pausePlayBtn.innerHTML = '<i class="fas fa-pause"></i><span id="control-text">Pause</span>';
}

// ============================================================
// SEQUENTIAL QUEUE PLAYER
// items: Array<{ text:string, element:HTMLElement|null, revealFn?:()=>void }>
// revealFn — called right before the item's first chunk plays
//            (used to trigger key-point slide-in animation)
// onComplete — called when last chunk finishes
// ============================================================
async function playItems(items, onComplete) {
    stopAudio();

    // ── Pre-fetch all audio chunks ──
    const segments = [];
    for (const item of items) {
        const chunks = splitChunks(item.text);
        const audios = [];
        for (const chunk of chunks) {
            const a = await fetchChunkAudio(chunk);
            if (a) audios.push(a);
        }
        if (audios.length > 0) {
            segments.push({
                audios,
                element:  item.element  || null,
                revealFn: item.revealFn || null
            });
        }
    }

    if (segments.length === 0) {
        if (onComplete) onComplete();
        return;
    }

    isSpeaking = true;
    videoPlay();

    let si = 0;   // segment index
    let ci = 0;   // chunk index within segment

    function next() {
        if (isPaused) return;   // resume() will call next() again

        if (si >= segments.length) {
            // ── All done ──
            isSpeaking = false;
            isPaused   = false;
            clearActive();
            videoPause();
            currentAudio = null;
            if (onComplete) onComplete();
            return;
        }

        const seg   = segments[si];
        const audio = seg.audios[ci];

        if (ci === 0) {
            // First chunk of this segment — reveal bullet + highlight
            if (seg.revealFn) seg.revealFn();
            setActive(seg.element);
        }

        currentAudio       = audio;
        audio.currentTime  = 0;

        audio.play().catch(() => advance());
        audio.onended = () => advance();
    }

    function advance() {
        ci++;
        if (ci >= segments[si].audios.length) { si++; ci = 0; }
        next();
    }

    // ── Wire pause/play button to this session ──
    window.__ttsResume = () => next();

    next();
}

// ============================================================
// UI CONTROLS
// ============================================================

// Menu
menuToggle.addEventListener('click', () => {
    hamburgerMenu.classList.add('active');
    menuOverlay.classList.add('active');
});
closeMenuBtn.addEventListener('click', () => {
    hamburgerMenu.classList.remove('active');
    menuOverlay.classList.remove('active');
});
menuOverlay.addEventListener('click', () => {
    hamburgerMenu.classList.remove('active');
    menuOverlay.classList.remove('active');
});

// Pause / Play — works with gTTS audio
pausePlayBtn.addEventListener('click', () => {
    if (!isSpeaking && !isPaused) return;

    if (isPaused) {
        isPaused = false;
        pausePlayBtn.innerHTML = '<i class="fas fa-pause"></i><span id="control-text">Pause</span>';
        if (currentAudio) {
            currentAudio.play().catch(() => {});
            videoPlay();
        } else if (window.__ttsResume) {
            window.__ttsResume();   // restart queue from where it stopped
        }
    } else {
        isPaused = true;
        pausePlayBtn.innerHTML = '<i class="fas fa-play"></i><span id="control-text">Play</span>';
        if (currentAudio) {
            currentAudio.pause();
            videoPause();
        }
    }
});

// Input method tabs
methodTabs.forEach(tab => {
    tab.addEventListener('click', () => {
        methodTabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        if (tab.dataset.method === 'type') {
            typeInput.classList.remove('hidden');
            speakInput.classList.add('hidden');
        } else {
            speakInput.classList.remove('hidden');
            typeInput.classList.add('hidden');
        }
    });
});

// Voice recording
voiceRecordBtn.addEventListener('click', () => {
    if (!recognition) { alert('Speech recognition not supported'); return; }

    if (!isRecording) {
        recognition.start();
        isRecording = true;
        voiceRecordBtn.classList.add('recording');
        recordStatus.textContent = 'Listening… Speak now';
        recognition.onresult = e => { doubtText.value = e.results[0][0].transcript; };
        recognition.onerror  = recognition.onend = () => {
            isRecording = false;
            voiceRecordBtn.classList.remove('recording');
            recordStatus.textContent = 'Click to start recording';
        };
    } else {
        recognition.stop();
    }
});

// ============================================================
// TOPIC DISPLAY
// Mirrors step_4 loop:
//  1. Print / speak explanation
//  2. Print / speak each key point  (bullet appears as voice reads it)
//  3. Print / speak example
//  4. Pause → ask "do you have doubts?"
// ============================================================
async function loadTopics() {
    try {
        const res  = await fetch('/api/get-topics');
        const data = await res.json();
        if (data.status === 'success') {
            topics = data.topics;
            displayTopic(0);
        } else {
            alert('Error loading lesson content');
        }
    } catch (e) {
        alert('Error loading lesson content');
    }
}

function displayTopic(index) {
    if (index >= topics.length) {
        window.location.href = '/quiz';
        return;
    }

    stopAudio();

    currentTopicIndex = index;
    const topic       = topics[index];

    // ── Topic header ──
    topicNumberEl.textContent = `Topic ${index + 1} of ${topics.length}`;
    topicTitleEl.textContent  = `Topic ${index + 1}`;

    // ── Explanation ──
    explanationText.innerHTML = '';
    let explainEl = null;
    if (topic.clean_explanation) {
        const p       = document.createElement('p');
        p.textContent = topic.clean_explanation;
        explanationText.appendChild(p);
        explainEl = p;
        accumulatedContext += '\n\n' + topic.clean_explanation;
    }

    // ── Key points — all hidden, revealed one-by-one as voice reads each ──
    keyPointsEl.innerHTML = '';
    const kpEls = [];
    if (topic.key_points && topic.key_points.length > 0) {
        topic.key_points.forEach(point => {
            const li       = document.createElement('li');
            li.textContent = point;
            // Initially invisible (we'll trigger animation via revealFn)
            li.style.opacity   = '0';
            li.style.transform = 'translateX(-20px)';
            li.style.animation = 'none';
            keyPointsEl.appendChild(li);
            kpEls.push(li);
        });
    }

    // ── Example ──
    let exampleEl = null;
    if (topic.example && topic.example.trim()) {
        exampleBox.style.display  = 'block';
        exampleTextEl.textContent = topic.example;
        exampleEl = exampleBox;
    } else {
        exampleBox.style.display = 'none';
    }

    // ── Reset interaction panel ──
    understandingCheck.classList.add('hidden');
    doubtSection.classList.add('hidden');
    answerBox.classList.add('hidden');
    doubtText.value = '';
    isPaused        = false;
    pausePlayBtn.innerHTML = '<i class="fas fa-pause"></i><span id="control-text">Pause</span>';
    chalkboardSection.scrollTop = 0;

    // ── Assemble TTS item list ──
    const items = [];

    // 1. Explanation
    if (topic.clean_explanation && explainEl) {
        items.push({ text: topic.clean_explanation, element: explainEl });
    }

    // 2. Key points — each bullet revealed right before its audio starts
    kpEls.forEach((li, i) => {
        items.push({
            text:     topic.key_points[i],
            element:  li,
            revealFn: () => revealBullet(li)   // triggers CSS slide-in
        });
    });

    // 3. Example
    if (topic.example && topic.example.trim() && exampleEl) {
        items.push({ text: `For example. ${topic.example}`, element: exampleEl });
    }

    // ── Play and then show Q&A prompt (step_4: "do you have doubts?") ──
    playItems(items, () => {
        setTimeout(() => {
            understandingCheck.classList.remove('hidden');
            understandingCheck.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }, 500);
    });
}

/** Trigger existing CSS slide-in animation for a key-point bullet */
function revealBullet(li) {
    li.style.animation = '';
    li.style.opacity   = '0';
    li.style.transform = 'translateX(-20px)';
    void li.offsetWidth;                          // force reflow
    li.style.animation = 'slideInPoint 0.5s ease forwards';
}

// ── Yes / No (step_4: no → next topic, yes → ask question) ───
btnYes.addEventListener('click', () => {
    stopAudio();
    displayTopic(currentTopicIndex + 1);
});

btnNo.addEventListener('click', () => {
    stopAudio();
    understandingCheck.classList.add('hidden');
    doubtSection.classList.remove('hidden');
    setTimeout(() => {
        doubtSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 300);
});

// ── Submit Doubt (step_4: answer printed + spoken) ────────────
submitDoubt.addEventListener('click', async () => {
    const question = doubtText.value.trim();
    if (!question) { alert('Please enter your question'); return; }

    answerLoading.classList.remove('hidden');
    stopAudio();

    try {
        const res  = await fetch('/api/answer-doubt', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ question, context: accumulatedContext })
        });
        const data = await res.json();
        answerLoading.classList.add('hidden');

        if (data.status === 'success') {
            const p        = document.createElement('p');
            p.style.lineHeight = '1.8';
            p.style.fontSize   = '1.125rem';
            p.textContent      = data.answer;
            answerTextEl.innerHTML = '';
            answerTextEl.appendChild(p);
            answerBox.classList.remove('hidden');

            setTimeout(() => {
                answerBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }, 300);

            // Speak the tutor answer in Indian female voice
            playItems([{ text: data.answer, element: p }], null);
        } else {
            alert('Error: ' + data.message);
        }
    } catch (err) {
        answerLoading.classList.add('hidden');
        alert('Error submitting question');
    }
});

// ── Continue Learning ─────────────────────────────────────────
continueBtn.addEventListener('click', () => {
    stopAudio();
    displayTopic(currentTopicIndex + 1);
});

// ── Cleanup ───────────────────────────────────────────────────
window.addEventListener('beforeunload', () => stopAudio());

// ── Boot ──────────────────────────────────────────────────────
loadTopics();