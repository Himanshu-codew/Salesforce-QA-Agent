/**
 * Salesforce Copilot — Chat UI Logic
 * WebSocket client with real-time tool execution display,
 * streaming text rendering, theme switching, and accessibility.
 */

// ─── SVG Icon Library ───
const SVG_ICONS = {
    user: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="4"/><path d="M20 21a8 8 0 1 0-16 0"/></svg>',
    bot: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="8.5" cy="16" r="1.5"/><circle cx="15.5" cy="16" r="1.5"/><path d="M12 2v4"/><path d="M9 5h6"/></svg>',
    copy: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>',
    check: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
    fileDoc: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>',
    fileSpreadsheet: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="8" y1="13" x2="16" y2="13"/><line x1="8" y1="17" x2="16" y2="17"/><line x1="12" y1="11" x2="12" y2="19"/></svg>',
    filePdf: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><path d="M10 12h1.5a1.5 1.5 0 0 1 0 3H10v3"/></svg>',
    fileImage: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>',
    fileText: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>',
    loading: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>',
    error: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
    tool: '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>',
    success: '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
    sparkle: '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.8 4.2L18 9l-4.2 1.8L12 15l-1.8-4.2L6 9l4.2-1.8L12 3z"/></svg>',
};

// ─── State ───
let ws = null;
let isConnected = false;
let isProcessing = false;
let currentAttachedFile = null;
const sessionId = crypto.randomUUID();

// ─── DOM Elements ───
const messagesContainer = document.getElementById('messagesContainer');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const clearChatBtn = document.getElementById('clearChatBtn');
const newChatBtn = document.getElementById('newChatBtn');
const connectionStatus = document.getElementById('connectionStatus');
const headerSubtitle = document.getElementById('headerSubtitle');
const welcomeScreen = document.getElementById('welcomeScreen');
const charCount = document.getElementById('charCount');
const mobileMenuBtn = document.getElementById('mobileMenuBtn');
const sidebar = document.getElementById('sidebar');
const fileInput = document.getElementById('fileInput');
const attachBtn = document.getElementById('attachBtn');
const filePreviewContainer = document.getElementById('filePreviewContainer');
const previewFileIcon = document.getElementById('previewFileIcon');
const previewFileName = document.getElementById('previewFileName');
const previewFileSize = document.getElementById('previewFileSize');
const removeFileBtn = document.getElementById('removeFileBtn');
const inputArea = document.getElementById('inputArea');
const themeToggle = document.getElementById('themeToggle');
const themeColorMeta = document.getElementById('themeColorMeta');

// ─── Theme Management ───
const THEME_KEY = 'sf-copilot-theme';

function getSystemTheme() {
    return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

function getInitialTheme() {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved === 'light' || saved === 'dark') return saved;
    return getSystemTheme();
}

function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem(THEME_KEY, theme);
    if (themeToggle) themeToggle.setAttribute('aria-pressed', theme === 'light');
    if (themeColorMeta) themeColorMeta.setAttribute('content', theme === 'dark' ? '#0B0814' : '#F7F5FC');
}

function setupTheme() {
    applyTheme(getInitialTheme());

    themeToggle.addEventListener('click', () => {
        const current = document.documentElement.getAttribute('data-theme');
        applyTheme(current === 'dark' ? 'light' : 'dark');
    });

    // Follow system changes when no explicit preference saved
    window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', (e) => {
        const saved = localStorage.getItem(THEME_KEY);
        if (!saved || (saved !== 'light' && saved !== 'dark')) {
            applyTheme(e.matches ? 'light' : 'dark');
        }
    });
}

// ─── Initialize ───
document.addEventListener('DOMContentLoaded', () => {
    setupTheme();
    connectWebSocket();
    setupEventListeners();
    setupFileUpload();
    setupSidebarBackdrop();
    messageInput.focus();
});

// ─── File Upload Helpers ───
function getFileIconSvg(filename) {
    const ext = (filename || '').split('.').pop().toLowerCase();
    if (['csv', 'xlsx', 'xls'].includes(ext)) return SVG_ICONS.fileSpreadsheet;
    if (['pdf'].includes(ext)) return SVG_ICONS.filePdf;
    if (['png', 'jpg', 'jpeg', 'gif', 'webp'].includes(ext)) return SVG_ICONS.fileImage;
    if (['txt', 'json', 'md', 'xml', 'log'].includes(ext)) return SVG_ICONS.fileText;
    return SVG_ICONS.fileDoc;
}

function formatFileSize(bytes) {
    if (!bytes) return '0 B';
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function setupFileUpload() {
    if (!attachBtn || !fileInput) return;

    attachBtn.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) uploadFile(file);
    });

    if (removeFileBtn) {
        removeFileBtn.addEventListener('click', clearAttachedFile);
    }

    if (inputArea) {
        let dragDepth = 0;
        ['dragenter', 'dragover'].forEach(evtName => {
            inputArea.addEventListener(evtName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dragDepth++;
                inputArea.classList.add('drag-over');
            });
        });

        ['dragleave'].forEach(evtName => {
            inputArea.addEventListener(evtName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dragDepth = Math.max(0, dragDepth - 1);
                if (dragDepth === 0) inputArea.classList.remove('drag-over');
            });
        });

        inputArea.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dragDepth = 0;
            inputArea.classList.remove('drag-over');
            const file = e.dataTransfer.files[0];
            if (file) uploadFile(file);
        });
    }
}

async function uploadFile(file) {
    if (!filePreviewContainer) return;

    filePreviewContainer.style.display = 'flex';
    previewFileIcon.innerHTML = SVG_ICONS.loading;
    previewFileName.textContent = `Uploading ${file.name}...`;
    previewFileSize.textContent = formatFileSize(file.size);
    attachBtn.classList.add('has-file');

    const formData = new FormData();
    formData.append('file', file);

    const startTime = Date.now();
    let uploadDone = false;

    const timerInterval = setInterval(() => {
        if (uploadDone) { clearInterval(timerInterval); return; }
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
        previewFileName.textContent = `Processing ${file.name}... (${elapsed}s)`;
    }, 500);

    try {
        const resp = await fetch('/upload', { method: 'POST', body: formData });
        uploadDone = true;
        clearInterval(timerInterval);

        if (!resp.ok) throw new Error(`Upload failed with status ${resp.status}`);

        const data = await resp.json();
        currentAttachedFile = data;

        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        previewFileIcon.innerHTML = getFileIconSvg(file.name);
        previewFileName.textContent = file.name;
        previewFileSize.textContent = `${formatFileSize(file.size)} · ${elapsed}s`;
        sendBtn.disabled = false;
    } catch (err) {
        uploadDone = true;
        clearInterval(timerInterval);
        console.error('File upload error:', err);
        previewFileName.textContent = 'Upload failed: ' + err.message;
        previewFileIcon.innerHTML = SVG_ICONS.error;
        currentAttachedFile = null;
        setTimeout(clearAttachedFile, 3000);
    }
}

function clearAttachedFile() {
    currentAttachedFile = null;
    if (fileInput) fileInput.value = '';
    if (filePreviewContainer) filePreviewContainer.style.display = 'none';
    if (attachBtn) attachBtn.classList.remove('has-file');
    sendBtn.disabled = !messageInput.value.trim();
}

// ─── WebSocket Connection ───
function connectWebSocket() {
    updateConnectionStatus('connecting');

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/${sessionId}`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        isConnected = true;
        updateConnectionStatus('connected');
        headerSubtitle.textContent = 'Connected — Ready to help';
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleServerEvent(data);
        } catch (e) {
            console.error('Failed to parse message:', e);
        }
    };

    ws.onclose = () => {
        isConnected = false;
        updateConnectionStatus('disconnected');
        headerSubtitle.textContent = 'Disconnected — Reconnecting...';
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateConnectionStatus('disconnected');
    };
}

function updateConnectionStatus(status) {
    if (!connectionStatus) return;
    const dot = connectionStatus.querySelector('.status-dot');
    const text = connectionStatus.querySelector('.status-text');

    dot.className = 'status-dot ' + status;
    const labels = {
        connected: 'Connected',
        disconnected: 'Disconnected',
        connecting: 'Connecting...',
    };
    text.textContent = labels[status] || status;
    dot.setAttribute('aria-label', labels[status] || status);
}

// ─── Event Listeners ───
function setupEventListeners() {
    sendBtn.addEventListener('click', sendMessage);

    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = Math.min(messageInput.scrollHeight, 140) + 'px';
        sendBtn.disabled = !messageInput.value.trim() && !currentAttachedFile;
        charCount.textContent = `${messageInput.value.length} / 4000`;
    });

    clearChatBtn.addEventListener('click', clearChat);

    newChatBtn.addEventListener('click', clearChat);

    document.querySelectorAll('.quick-action-btn, .example-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const query = btn.dataset.query;
            if (query) {
                messageInput.value = query;
                messageInput.dispatchEvent(new Event('input'));
                sendMessage();
            }
        });
    });

    mobileMenuBtn.addEventListener('click', () => {
        sidebar.classList.toggle('open');
        toggleSidebarBackdrop(true);
    });

    messagesContainer.addEventListener('click', closeSidebar);
}

function setupSidebarBackdrop() {
    const backdrop = document.createElement('div');
    backdrop.className = 'sidebar-backdrop';
    backdrop.id = 'sidebarBackdrop';
    document.body.appendChild(backdrop);
    backdrop.addEventListener('click', closeSidebar);
}

function toggleSidebarBackdrop(show) {
    const backdrop = document.getElementById('sidebarBackdrop');
    if (backdrop) backdrop.classList.toggle('show', show);
}

function closeSidebar() {
    sidebar.classList.remove('open');
    toggleSidebarBackdrop(false);
}

// ─── Send Message ───
function sendMessage() {
    const text = messageInput.value.trim();
    const attachedFile = currentAttachedFile;

    if ((!text && !attachedFile) || !isConnected || isProcessing) return;

    if (welcomeScreen) welcomeScreen.style.display = 'none';
    closeSidebar();

    appendMessage('user', text, attachedFile);

    ws.send(JSON.stringify({
        type: 'message',
        content: text,
        file_info: attachedFile,
    }));

    messageInput.value = '';
    messageInput.style.height = 'auto';
    clearAttachedFile();
    sendBtn.disabled = true;
    charCount.textContent = '0 / 4000';
    isProcessing = true;
    headerSubtitle.textContent = 'Processing...';
}

// ─── Handle Server Events ───
function handleServerEvent(event) {
    switch (event.type) {
        case 'thinking':
            appendThinking(event.data);
            break;

        case 'tool_call':
            removeThinking();
            appendThinking(`Accessing Salesforce (${event.data.name})...`);
            break;

        case 'tool_result':
            removeThinking();
            appendThinking('Analyzing data & generating response...');
            break;

        case 'response':
            removeThinking();
            appendMessage('assistant', event.data, null, true);
            isProcessing = false;
            headerSubtitle.textContent = 'Ready to help';
            break;

        case 'confirmation':
            removeThinking();
            appendConfirmation(event.data);
            isProcessing = false;
            headerSubtitle.textContent = 'Awaiting confirmation...';
            break;

        case 'error':
            removeThinking();
            appendMessage('assistant', event.data);
            isProcessing = false;
            headerSubtitle.textContent = 'Ready to help';
            break;

        default:
            console.warn('Unknown event type:', event.type);
    }
}

// ─── UI Rendering ───
function appendMessage(role, content, attachedFile = null, stream = false) {
    const messageEl = document.createElement('div');
    messageEl.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.setAttribute('aria-hidden', 'true');
    avatar.innerHTML = role === 'user' ? SVG_ICONS.user : SVG_ICONS.bot;

    const contentEl = document.createElement('div');
    contentEl.className = 'message-content';

    if (attachedFile) {
        const fileTag = document.createElement('div');
        fileTag.className = 'message-attachment-tag';
        const iconSvg = getFileIconSvg(attachedFile.filename || '');
        const sizeStr = formatFileSize(attachedFile.file_size || 0);
        fileTag.innerHTML = `${iconSvg} <span>${escapeHtml(attachedFile.filename)}</span> <span style="opacity:0.65; font-size:0.65rem;">(${sizeStr})</span>`;
        contentEl.appendChild(fileTag);
    }

    const textWrapper = document.createElement('div');
    textWrapper.className = 'text-wrapper';
    contentEl.appendChild(textWrapper);

    if (role === 'assistant') {
        const copyBtn = document.createElement('button');
        copyBtn.className = 'copy-response-btn';
        copyBtn.title = 'Copy response text';
        copyBtn.innerHTML = `${SVG_ICONS.copy} Copy`;
        copyBtn.addEventListener('click', () => {
            const textToCopy = contentEl.innerText
                .replace(/Copy$/, '')
                .replace(/Copied!$/, '')
                .trim();
            navigator.clipboard.writeText(textToCopy).then(() => {
                copyBtn.innerHTML = `${SVG_ICONS.check} Copied!`;
                copyBtn.classList.add('copied');
                setTimeout(() => {
                    copyBtn.innerHTML = `${SVG_ICONS.copy} Copy`;
                    copyBtn.classList.remove('copied');
                }, 2000);
            });
        });
        contentEl.appendChild(copyBtn);
    }

    messageEl.appendChild(avatar);
    messageEl.appendChild(contentEl);
    messagesContainer.appendChild(messageEl);

    if (content) {
        if (stream) {
            streamMarkdown(textWrapper, content);
        } else {
            textWrapper.innerHTML = renderMarkdown(content);
        }
    }

    scrollToBottom();
}

/**
 * Stream markdown into the container progressively (token-by-token).
 * Falls back to instant render when reduced motion is preferred.
 */
function streamMarkdown(container, markdown) {
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduceMotion || markdown.length < 40) {
        container.innerHTML = renderMarkdown(markdown);
        return;
    }

    const fullHtml = renderMarkdown(markdown);
    const plain = markdown;
    let index = 0;
    const chunkSize = 4;
    let renderedPlain = '';

    // Set aria-busy while streaming
    const messageEl = container.closest('.message');
    if (messageEl) messageEl.setAttribute('aria-busy', 'true');

    function tick() {
        if (index >= plain.length) {
            // Final render for proper formatting
            container.innerHTML = fullHtml;
            if (messageEl) messageEl.removeAttribute('aria-busy');
            scrollToBottom();
            return;
        }

        index = Math.min(index + chunkSize, plain.length);
        // Render progressively with full formatting each step
        container.innerHTML = renderMarkdown(plain.slice(0, index));
        scrollToBottom();
        setTimeout(tick, 16);
    }

    tick();
}

function appendThinking(text) {
    removeThinking();

    const el = document.createElement('div');
    el.className = 'thinking-indicator';
    el.id = 'thinkingIndicator';
    el.setAttribute('role', 'status');

    el.innerHTML = `
        <div class="spinner" aria-hidden="true">
            <span class="dot"></span>
            <span class="dot"></span>
            <span class="dot"></span>
        </div>
        <span class="thinking-text">${escapeHtml(text)}</span>
    `;

    messagesContainer.appendChild(el);
    scrollToBottom();
}

function removeThinking() {
    const el = document.getElementById('thinkingIndicator');
    if (el) el.remove();
}

function appendConfirmation(content) {
    const el = document.createElement('div');
    el.className = 'confirmation-event';

    const contentEl = document.createElement('div');
    contentEl.className = 'message-content';
    contentEl.innerHTML = renderMarkdown(content);

    el.appendChild(contentEl);
    messagesContainer.appendChild(el);
    scrollToBottom();
}

function clearChat() {
    const messages = messagesContainer.querySelectorAll(
        '.message, .tool-event, .thinking-indicator, .confirmation-event'
    );
    messages.forEach(el => el.remove());

    if (welcomeScreen) welcomeScreen.style.display = '';

    if (ws && isConnected) {
        ws.send(JSON.stringify({ type: 'clear' }));
    }

    messageInput.focus();
}

// ─── Helpers ───
function scrollToBottom() {
    requestAnimationFrame(() => {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Simple markdown renderer — handles common patterns
 * without a full library dependency.
 */
function renderMarkdown(text) {
    if (!text) return '';

    let html = escapeHtml(text);

    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
        return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
    });

    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    html = html.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, '<em>$1</em>');

    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');

    html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

    html = html.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');

    html = html.replace(/^---$/gm, '<hr>');

    html = renderTables(html);

    html = html.replace(/\n\n/g, '</p><p>');
    html = html.replace(/\n/g, '<br>');
    html = '<p>' + html + '</p>';

    html = html.replace(/<p>\s*<\/p>/g, '');
    html = html.replace(/<p>\s*(<h[123]>)/g, '$1');
    html = html.replace(/(<\/h[123]>)\s*<\/p>/g, '$1');
    html = html.replace(/<p>\s*(<pre>)/g, '$1');
    html = html.replace(/(<\/pre>)\s*<\/p>/g, '$1');
    html = html.replace(/<p>\s*(<ul>)/g, '$1');
    html = html.replace(/(<\/ul>)\s*<\/p>/g, '$1');
    html = html.replace(/<p>\s*(<table>)/g, '$1');
    html = html.replace(/(<\/table>)\s*<\/p>/g, '$1');
    html = html.replace(/<p>\s*(<hr>)\s*<\/p>/g, '$1');

    return html;
}

/**
 * Render pipe-delimited tables into HTML tables.
 */
function renderTables(html) {
    const lines = html.split('\n');
    const result = [];
    let inTable = false;
    let tableRows = [];

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();

        if (line.startsWith('|') && line.endsWith('|')) {
            if (/^\|[\s\-:]+\|/.test(line) && line.includes('---')) {
                continue;
            }

            if (!inTable) {
                inTable = true;
                tableRows = [];
            }

            const cells = line
                .slice(1, -1)
                .split('|')
                .map(c => c.trim());

            tableRows.push(cells);
        } else {
            if (inTable) {
                result.push(buildTable(tableRows));
                inTable = false;
                tableRows = [];
            }
            result.push(lines[i]);
        }
    }

    if (inTable) result.push(buildTable(tableRows));

    return result.join('\n');
}

function buildTable(rows) {
    if (rows.length === 0) return '';

    let html = '<table>';
    html += '<thead><tr>';
    rows[0].forEach(cell => {
        html += `<th>${cell}</th>`;
    });
    html += '</tr></thead>';

    if (rows.length > 1) {
        html += '<tbody>';
        for (let i = 1; i < rows.length; i++) {
            html += '<tr>';
            rows[i].forEach(cell => {
                html += `<td>${cell}</td>`;
            });
            html += '</tr>';
        }
        html += '</tbody>';
    }

    html += '</table>';
    return html;
}

// ─── Salesforce Direct Login Modal Logic ───
const connectSfBtn = document.getElementById('connectSfBtn');
const userProfileBadge = document.getElementById('userProfileBadge');
const userDisplayName = document.getElementById('userDisplayName');
const logoutBtn = document.getElementById('logoutBtn');
const sfConnectModal = document.getElementById('sfConnectModal');
const closeModalBtn = document.getElementById('closeModalBtn');
const submitDirectPassBtn = document.getElementById('submitDirectPassBtn');
const modalStatusMsg = document.getElementById('modalStatusMsg');

function showModalMsg(text, type = 'error') {
    if (!modalStatusMsg) return;
    modalStatusMsg.innerHTML = type === 'error'
        ? `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg> ${text}`
        : `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg> ${text}`;
    modalStatusMsg.className = `modal-status-msg ${type}`;
    modalStatusMsg.classList.remove('hidden');

    if (type === 'error' && sfConnectModal) {
        const card = sfConnectModal.querySelector('.modal-card');
        if (card) {
            card.classList.remove('shake');
            void card.offsetWidth; // Trigger reflow
            card.classList.add('shake');
            setTimeout(() => card.classList.remove('shake'), 450);
        }
    }
}

function hideModalMsg() {
    if (modalStatusMsg) modalStatusMsg.classList.add('hidden');
}

function openConnectModal() {
    hideModalMsg();
    if (sfConnectModal) sfConnectModal.classList.remove('hidden');
    const userField = document.getElementById('directUsername');
    if (userField) userField.focus();
}

function closeConnectModal() {
    if (sfConnectModal) sfConnectModal.classList.add('hidden');
}

async function checkUserAuthStatus() {
    try {
        const res = await fetch(`/api/auth/me?session_id=${sessionId}`);
        if (!res.ok) return;
        const data = await res.json();

        if (data && data.authenticated) {
            if (connectSfBtn) connectSfBtn.style.display = 'none';
            if (userProfileBadge) userProfileBadge.classList.remove('hidden');
            if (userDisplayName) {
                const user = data.user || {};
                userDisplayName.textContent = `${user.display_name || 'Salesforce User'} (${user.org_name || 'Connected'})`;
            }
        } else {
            if (connectSfBtn) connectSfBtn.style.display = 'inline-flex';
            if (userProfileBadge) userProfileBadge.classList.add('hidden');
        }
    } catch (e) {
        console.warn('Auth status check notice:', e);
    }
}

async function handleDirectPasswordConnect() {
    hideModalMsg();
    const usernameInput = document.getElementById('directUsername');
    const passwordInput = document.getElementById('directPassword');
    const secTokenInput = document.getElementById('directSecToken');
    const domainSelect = document.getElementById('directDomainSelect');

    const username = usernameInput ? usernameInput.value.trim() : '';
    const password = passwordInput ? passwordInput.value.trim() : '';
    const security_token = secTokenInput ? secTokenInput.value.trim() : '';
    const domain = domainSelect ? domainSelect.value : 'login';

    if (!username || !password) {
        showModalMsg('Please enter your Salesforce Username and Password.', 'error');
        return;
    }

    const origBtnContent = submitDirectPassBtn ? submitDirectPassBtn.innerHTML : '';
    if (submitDirectPassBtn) {
        submitDirectPassBtn.disabled = true;
        submitDirectPassBtn.innerHTML = `
            <svg class="spin-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="animation: spin 0.8s linear infinite;">
                <circle cx="12" cy="12" r="10" stroke-dasharray="32" stroke-dashoffset="12"/>
            </svg>
            <span>Authenticating...</span>
        `;
    }

    try {
        const res = await fetch('/api/auth/connect_direct', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                mode: 'password',
                username,
                password,
                security_token,
                domain
            })
        });

        const data = await res.json();
        if (res.ok && data.success) {
            showModalMsg('Connected successfully to Salesforce!', 'success');
            await checkUserAuthStatus();
            setTimeout(() => {
                closeConnectModal();
            }, 800);
        } else {
            showModalMsg(data.error || 'Authentication failed. Check your username & password.', 'error');
        }
    } catch (err) {
        showModalMsg(`Connection error: ${err.message}`, 'error');
    } finally {
        if (submitDirectPassBtn) {
            submitDirectPassBtn.disabled = false;
            submitDirectPassBtn.innerHTML = origBtnContent;
        }
    }
}

async function logoutSfUser() {
    try {
        await fetch(`/api/auth/logout?session_id=${sessionId}`, { method: 'POST' });
        await checkUserAuthStatus();
        if (typeof addSystemMessage === 'function') {
            addSystemMessage('🔒 Disconnected from Salesforce. You are back on default connection.');
        }
    } catch (e) {
        console.error('Logout error:', e);
    }
}

// Event Listeners
if (connectSfBtn) connectSfBtn.addEventListener('click', openConnectModal);
if (closeModalBtn) closeModalBtn.addEventListener('click', closeConnectModal);
if (submitDirectPassBtn) submitDirectPassBtn.addEventListener('click', handleDirectPasswordConnect);
if (logoutBtn) logoutBtn.addEventListener('click', logoutSfUser);

// Allow Enter key submission in password field
const directPasswordElem = document.getElementById('directPassword');
if (directPasswordElem) {
    directPasswordElem.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            handleDirectPasswordConnect();
        }
    });
}

// Run auth check on initialization
document.addEventListener('DOMContentLoaded', checkUserAuthStatus);




