/**
 * Salesforce AI Agent — Chat UI Logic
 * WebSocket client with real-time tool execution display,
 * markdown rendering, and interactive chat.
 */

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

// ─── Initialize ───
document.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
    setupEventListeners();
    setupFileUpload();
    messageInput.focus();
});

// ─── File Upload Helpers ───
function getFileIcon(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    if (['csv', 'xlsx', 'xls'].includes(ext)) return '📊';
    if (['pdf'].includes(ext)) return '📕';
    if (['png', 'jpg', 'jpeg', 'gif', 'webp'].includes(ext)) return '🖼️';
    if (['txt', 'json', 'md', 'xml', 'log'].includes(ext)) return '📝';
    return '📄';
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
        if (file) {
            uploadFile(file);
        }
    });

    if (removeFileBtn) {
        removeFileBtn.addEventListener('click', clearAttachedFile);
    }

    // Drag and drop support
    if (inputArea) {
        ['dragenter', 'dragover'].forEach(evtName => {
            inputArea.addEventListener(evtName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                inputArea.classList.add('drag-over');
            });
        });

        ['dragleave', 'drop'].forEach(evtName => {
            inputArea.addEventListener(evtName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                inputArea.classList.remove('drag-over');
            });
        });

        inputArea.addEventListener('drop', (e) => {
            const file = e.dataTransfer.files[0];
            if (file) {
                uploadFile(file);
            }
        });
    }
}

async function uploadFile(file) {
    if (!filePreviewContainer) return;

    filePreviewContainer.style.display = 'flex';
    previewFileIcon.textContent = '⏳';
    previewFileName.textContent = `Uploading ${file.name}...`;
    previewFileSize.textContent = formatFileSize(file.size);
    attachBtn.classList.add('has-file');

    const formData = new FormData();
    formData.append('file', file);

    const startTime = Date.now();
    let uploadDone = false;

    // Show elapsed time while processing
    const timerInterval = setInterval(() => {
        if (uploadDone) { clearInterval(timerInterval); return; }
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
        if (!uploadDone) {
            previewFileName.textContent = `Processing ${file.name}... (${elapsed}s)`;
        }
    }, 500);

    try {
        const resp = await fetch('/upload', {
            method: 'POST',
            body: formData,
        });

        uploadDone = true;
        clearInterval(timerInterval);

        if (!resp.ok) {
            throw new Error(`Upload failed with status ${resp.status}`);
        }

        const data = await resp.json();
        currentAttachedFile = data;

        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        previewFileIcon.textContent = getFileIcon(file.name);
        previewFileName.textContent = file.name;
        previewFileSize.textContent = `${formatFileSize(file.size)} • ${elapsed}s`;
        sendBtn.disabled = false;
    } catch (err) {
        uploadDone = true;
        clearInterval(timerInterval);
        console.error('File upload error:', err);
        previewFileName.textContent = 'Upload failed: ' + err.message;
        previewFileIcon.textContent = '❌';
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
        // Auto-reconnect after 3s
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateConnectionStatus('disconnected');
    };
}

function updateConnectionStatus(status) {
    const dot = connectionStatus.querySelector('.status-dot');
    const text = connectionStatus.querySelector('.status-text');

    dot.className = 'status-dot ' + status;
    const labels = {
        connected: 'Connected',
        disconnected: 'Disconnected',
        connecting: 'Connecting...',
    };
    text.textContent = labels[status] || status;
}

// ─── Event Listeners ───
function setupEventListeners() {
    // Send message
    sendBtn.addEventListener('click', sendMessage);

    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
        sendBtn.disabled = !messageInput.value.trim() && !currentAttachedFile;
        charCount.textContent = `${messageInput.value.length} / 4000`;
    });

    // Clear chat
    clearChatBtn.addEventListener('click', clearChat);

    // Quick actions
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

    // Mobile menu
    mobileMenuBtn.addEventListener('click', () => {
        sidebar.classList.toggle('open');
    });

    // Close sidebar on message click (mobile)
    messagesContainer.addEventListener('click', () => {
        sidebar.classList.remove('open');
    });
}

// ─── Theme Toggle Logic (Global) ───
window.toggleTheme = function() {
    const isLight = document.body.classList.toggle('light-theme');
    localStorage.setItem('theme', isLight ? 'light' : 'dark');
    const icon = document.getElementById('themeIcon');
    const text = document.getElementById('themeText');
    if (icon) icon.textContent = isLight ? '☀️' : 'Light';
    if (text) text.textContent = isLight ? 'Light' : 'Dark';
};

// Auto-apply saved theme instantly on script load
(function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    if (savedTheme === 'light') {
        document.body.classList.add('light-theme');
        window.addEventListener('DOMContentLoaded', () => {
            const icon = document.getElementById('themeIcon');
            const text = document.getElementById('themeText');
            if (icon) icon.textContent = '☀️';
            if (text) text.textContent = 'Light';
        });
    }
})();

// ─── Send Message ───
function sendMessage() {
    const text = messageInput.value.trim();
    const attachedFile = currentAttachedFile;

    if ((!text && !attachedFile) || !isConnected || isProcessing) return;

    // Hide welcome screen
    if (welcomeScreen) {
        welcomeScreen.style.display = 'none';
    }

    // Add user message to UI with attachment if present
    appendMessage('user', text, attachedFile);

    // Send to server
    ws.send(JSON.stringify({
        type: 'message',
        content: text,
        file_info: attachedFile,
    }));

    // Clear input & attached file
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
            appendThinking(`⚡ Accessing Salesforce (${event.data.name})...`);
            break;

        case 'tool_result':
            removeThinking();
            appendThinking('🧠 Analyzing data & generating response...');
            break;

        case 'response':
            removeThinking();
            appendMessage('assistant', event.data);
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

function appendMessage(role, content, attachedFile = null) {
    const messageEl = document.createElement('div');
    messageEl.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = role === 'user' ? 'U' : '🤖';

    const contentEl = document.createElement('div');
    contentEl.className = 'message-content';

    // Render attachment badge for user if a file was attached
    if (attachedFile) {
        const fileTag = document.createElement('div');
        fileTag.className = 'message-attachment-tag';
        const icon = getFileIcon(attachedFile.filename || '');
        const sizeStr = formatFileSize(attachedFile.file_size || 0);
        fileTag.innerHTML = `<span>${icon}</span> <span>${escapeHtml(attachedFile.filename)}</span> <span style="opacity:0.75; font-size:0.7rem;">(${sizeStr})</span>`;
        contentEl.appendChild(fileTag);
    }

    if (content) {
        const textWrapper = document.createElement('div');
        textWrapper.innerHTML = renderMarkdown(content);
        contentEl.appendChild(textWrapper);
    }

    if (role === 'assistant') {
        const copyBtn = document.createElement('button');
        copyBtn.className = 'copy-response-btn';
        copyBtn.title = 'Copy response text';
        copyBtn.innerHTML = '📋 Copy';
        copyBtn.onclick = () => {
            // Strip HTML tags for clean text copy
            const textToCopy = contentEl.innerText.replace('📋 Copy', '').replace('✅ Copied!', '').trim();
            navigator.clipboard.writeText(textToCopy).then(() => {
                copyBtn.innerHTML = '✅ Copied!';
                copyBtn.classList.add('copied');
                setTimeout(() => {
                    copyBtn.innerHTML = '📋 Copy';
                    copyBtn.classList.remove('copied');
                }, 2000);
            });
        };
        contentEl.appendChild(copyBtn);
    }

    messageEl.appendChild(avatar);
    messageEl.appendChild(contentEl);
    messagesContainer.appendChild(messageEl);
    scrollToBottom();
}

function appendThinking(text) {
    removeThinking(); // Remove any existing

    const el = document.createElement('div');
    el.className = 'thinking-indicator';
    el.id = 'thinkingIndicator';

    el.innerHTML = `
        <div class="spinner">
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

function appendToolCall(data) {
    // Completely hidden from UI for clean user experience
}

function appendToolResult(data) {
    // Completely hidden from UI for clean user experience
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
    // Remove all messages but keep welcome screen
    const messages = messagesContainer.querySelectorAll(
        '.message, .tool-event, .thinking-indicator, .confirmation-event'
    );
    messages.forEach(el => el.remove());

    // Show welcome screen
    if (welcomeScreen) {
        welcomeScreen.style.display = '';
    }

    // Notify server
    if (ws && isConnected) {
        ws.send(JSON.stringify({ type: 'clear' }));
    }
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
 * Simple markdown renderer — handles the most common patterns
 * without a full library dependency.
 */
function renderMarkdown(text) {
    if (!text) return '';

    let html = escapeHtml(text);

    // Code blocks (```...```)
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
        return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
    });

    // Inline code (`...`)
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Bold (**...**)
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    // Italic (*...*)
    html = html.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, '<em>$1</em>');

    // Headers (## ...)
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    // Unordered lists (- ...)
    html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');

    // Ordered lists (1. ...)
    html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

    // Links [text](url)
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');

    // Horizontal rules (---)
    html = html.replace(/^---$/gm, '<hr>');

    // Tables (basic | pipe tables)
    html = renderTables(html);

    // Line breaks → paragraphs
    html = html.replace(/\n\n/g, '</p><p>');
    html = html.replace(/\n/g, '<br>');
    html = '<p>' + html + '</p>';

    // Clean up empty paragraphs
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

    // Emojis for common patterns
    html = html.replace(/⚠️/g, '<span style="font-size:1.1em">⚠️</span>');
    html = html.replace(/✅/g, '<span style="font-size:1.1em">✅</span>');
    html = html.replace(/❌/g, '<span style="font-size:1.1em">❌</span>');

    return html;
}

/**
 * Render pipe-delimited tables into HTML tables.
 */
function renderTables(html) {
    const lines = html.split('\n');
    let result = [];
    let inTable = false;
    let tableRows = [];

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();

        if (line.startsWith('|') && line.endsWith('|')) {
            // Check if this is a separator line (|---|---|)
            if (/^\|[\s\-:]+\|/.test(line) && line.includes('---')) {
                // Skip separator line
                continue;
            }

            if (!inTable) {
                inTable = true;
                tableRows = [];
            }

            const cells = line
                .slice(1, -1)  // Remove leading/trailing |
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

    if (inTable) {
        result.push(buildTable(tableRows));
    }

    return result.join('\n');
}

function buildTable(rows) {
    if (rows.length === 0) return '';

    let html = '<table>';

    // First row as header
    html += '<thead><tr>';
    rows[0].forEach(cell => {
        html += `<th>${cell}</th>`;
    });
    html += '</tr></thead>';

    // Remaining rows
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
