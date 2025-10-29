// AI Coach Frontend Application
// Handles chat interface, API communication, and streaming responses

const API_BASE_URL = 'http://127.0.0.1:8030';
let isProcessing = false;
let currentConversationId = 'current';
let conversations = {};

// DOM Elements
const messagesContainer = document.getElementById('messagesContainer');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const conversationsList = document.getElementById('conversationsList');
const newChatBtn = document.getElementById('newChatBtn');

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();
});

/**
 * Initialize the application
 */
async function initializeApp() {
    await checkServerHealth();
    autoResizeTextarea();
    loadConversations();
    initializeCurrentConversation();
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Send button click
    sendButton.addEventListener('click', handleSendMessage);

    // Enter key to send (Shift+Enter for new line)
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });

    // Auto-resize textarea
    messageInput.addEventListener('input', autoResizeTextarea);

    // Enable/disable send button based on input
    messageInput.addEventListener('input', () => {
        const hasContent = messageInput.value.trim().length > 0;
        sendButton.disabled = !hasContent || isProcessing;
    });

    // New chat button
    newChatBtn.addEventListener('click', createNewConversation);

    // Conversation item clicks
    conversationsList.addEventListener('click', (e) => {
        const item = e.target.closest('.conversation-item');
        if (item) {
            const convId = item.dataset.conversationId;
            switchConversation(convId);
        }
    });
}

/**
 * Check server health status
 */
async function checkServerHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();

        if (data.status === 'healthy') {
            updateStatus('connected', 'Connected');
        } else {
            updateStatus('warning', 'Server issues detected');
        }
    } catch (error) {
        updateStatus('error', 'Cannot connect to server');
        console.error('Health check failed:', error);
    }
}

/**
 * Update connection status indicator
 */
function updateStatus(status, text) {
    statusDot.className = `status-dot ${status}`;
    statusText.textContent = text;
}

/**
 * Handle sending a message
 */
async function handleSendMessage() {
    const message = messageInput.value.trim();

    if (!message || isProcessing) return;

    // Add user message to chat
    addMessage(message, 'user');

    // Clear input
    messageInput.value = '';
    autoResizeTextarea();
    sendButton.disabled = true;
    isProcessing = true;

    // Show typing indicator
    const typingIndicator = addTypingIndicator();

    try {
        // Send message to API and handle streaming response
        await streamAgentResponse(message, typingIndicator);
    } catch (error) {
        console.error('Error sending message:', error);
        removeTypingIndicator(typingIndicator);
        addErrorMessage('Failed to get response from AI Coach. Please try again.');
    } finally {
        isProcessing = false;
        sendButton.disabled = messageInput.value.trim().length === 0;
    }
}

/**
 * Stream agent response from API
 */
async function streamAgentResponse(message, typingIndicator) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                conversation_id: null,
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Remove typing indicator
        removeTypingIndicator(typingIndicator);

        // Create message element for streaming response
        const assistantMessage = addMessage('', 'assistant', true);
        const messageContent = assistantMessage.querySelector('.message-content');

        // Read the streaming response
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();

            if (done) break;

            // Decode chunk
            buffer += decoder.decode(value, { stream: true });

            // Process complete SSE events
            const events = buffer.split('\n\n');
            buffer = events.pop() || ''; // Keep incomplete event in buffer

            for (const event of events) {
                if (!event.trim()) continue;

                // Parse SSE event
                const lines = event.split('\n');
                let eventData = null;

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.substring(6);
                        if (dataStr === '[DONE]') {
                            continue;
                        }
                        try {
                            eventData = JSON.parse(dataStr);
                        } catch (e) {
                            console.error('Failed to parse event data:', e);
                        }
                    }
                }

                if (eventData) {
                    // Handle different event types
                    if (eventData.type === 'token') {
                        // Append token to message
                        messageContent.textContent += eventData.content;
                        scrollToBottom();
                    } else if (eventData.type === 'complete') {
                        // Response complete
                        console.log('Response complete');
                    } else if (eventData.type === 'error') {
                        // Handle error
                        console.error('Stream error:', eventData.error);
                        messageContent.innerHTML = `<p style="color: var(--error);">Error: ${eventData.error}</p>`;
                    }
                }
            }
        }

        // Format the final message content (convert markdown-like formatting)
        formatMessageContent(messageContent);

        // Save conversation after receiving response
        saveCurrentMessages();
        renderConversationsList();

    } catch (error) {
        console.error('Streaming error:', error);
        throw error;
    }
}

/**
 * Add a message to the chat
 */
function addMessage(content, sender, isEmpty = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;

    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';

    if (!isEmpty) {
        messageContent.textContent = content;
    }

    messageDiv.appendChild(messageContent);
    messagesContainer.appendChild(messageDiv);

    scrollToBottom();

    return messageDiv;
}

/**
 * Add typing indicator
 */
function addTypingIndicator() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant-message';
    messageDiv.id = 'typing-indicator';

    const indicator = document.createElement('div');
    indicator.className = 'message-content typing-indicator';
    indicator.innerHTML = `
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
    `;

    messageDiv.appendChild(indicator);
    messagesContainer.appendChild(messageDiv);

    scrollToBottom();

    return messageDiv;
}

/**
 * Remove typing indicator
 */
function removeTypingIndicator(indicator) {
    if (indicator && indicator.parentNode) {
        indicator.parentNode.removeChild(indicator);
    }
}

/**
 * Add error message
 */
function addErrorMessage(error) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant-message';

    const messageContent = document.createElement('div');
    messageContent.className = 'message-content error-message';
    messageContent.textContent = error;

    messageDiv.appendChild(messageContent);
    messagesContainer.appendChild(messageDiv);

    scrollToBottom();
}

/**
 * Format message content (convert markdown-like text to HTML)
 */
function formatMessageContent(element) {
    let content = element.textContent;

    // Convert **bold** to <strong>
    content = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Convert URLs to links
    content = content.replace(
        /(https?:\/\/[^\s]+)/g,
        '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>'
    );

    // Convert line breaks
    content = content.replace(/\n/g, '<br>');

    element.innerHTML = content;
}

/**
 * Scroll to bottom of messages
 */
function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

/**
 * Auto-resize textarea based on content
 */
function autoResizeTextarea() {
    messageInput.style.height = 'auto';
    messageInput.style.height = messageInput.scrollHeight + 'px';
}

// Periodic health check (every 30 seconds)
setInterval(checkServerHealth, 30000);

// ==============================================================================
// Conversation Management
// ==============================================================================

/**
 * Load conversations from localStorage
 */
function loadConversations() {
    const saved = localStorage.getItem('ai_coach_conversations');
    if (saved) {
        try {
            conversations = JSON.parse(saved);
        } catch (e) {
            console.error('Failed to load conversations:', e);
            conversations = {};
        }
    }
    renderConversationsList();
}

/**
 * Save conversations to localStorage
 */
function saveConversations() {
    localStorage.setItem('ai_coach_conversations', JSON.stringify(conversations));
}

/**
 * Initialize current conversation if it doesn't exist
 */
function initializeCurrentConversation() {
    if (!conversations[currentConversationId]) {
        conversations[currentConversationId] = {
            id: currentConversationId,
            title: 'New Conversation',
            messages: [],
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString()
        };
        saveConversations();
    }
}

/**
 * Create a new conversation
 */
function createNewConversation() {
    const newId = 'conv_' + Date.now();
    conversations[newId] = {
        id: newId,
        title: 'New Conversation',
        messages: [],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
    };
    saveConversations();
    switchConversation(newId);
}

/**
 * Switch to a different conversation
 */
function switchConversation(conversationId) {
    // Save current conversation messages
    saveCurrentMessages();

    // Switch to new conversation
    currentConversationId = conversationId;

    // Clear and load messages
    clearMessages();
    loadConversationMessages(conversationId);

    // Update UI
    renderConversationsList();
}

/**
 * Save current conversation messages
 */
function saveCurrentMessages() {
    if (conversations[currentConversationId]) {
        const messages = [];
        const messageElements = messagesContainer.querySelectorAll('.message:not(.welcome-message)');

        messageElements.forEach(el => {
            const isUser = el.classList.contains('user-message');
            const content = el.querySelector('.message-content').textContent;
            messages.push({
                role: isUser ? 'user' : 'assistant',
                content: content,
                timestamp: new Date().toISOString()
            });
        });

        conversations[currentConversationId].messages = messages;
        conversations[currentConversationId].updatedAt = new Date().toISOString();

        // Update title based on first message if still "New Conversation"
        if (conversations[currentConversationId].title === 'New Conversation' && messages.length > 0) {
            const firstUserMsg = messages.find(m => m.role === 'user');
            if (firstUserMsg) {
                conversations[currentConversationId].title = firstUserMsg.content.substring(0, 40) + '...';
            }
        }

        saveConversations();
    }
}

/**
 * Clear messages from display
 */
function clearMessages() {
    const nonWelcomeMessages = messagesContainer.querySelectorAll('.message:not(.welcome-message)');
    nonWelcomeMessages.forEach(el => el.remove());
}

/**
 * Load conversation messages
 */
function loadConversationMessages(conversationId) {
    const conversation = conversations[conversationId];
    if (!conversation || !conversation.messages) return;

    conversation.messages.forEach(msg => {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${msg.role === 'user' ? 'user-message' : 'assistant-message'}`;

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.textContent = msg.content;

        messageDiv.appendChild(messageContent);
        messagesContainer.appendChild(messageDiv);
    });

    scrollToBottom();
}

/**
 * Render conversations list in sidebar
 */
function renderConversationsList() {
    conversationsList.innerHTML = '';

    // Sort conversations by updatedAt (most recent first)
    const sortedConvs = Object.values(conversations).sort((a, b) => {
        return new Date(b.updatedAt) - new Date(a.updatedAt);
    });

    sortedConvs.forEach(conv => {
        const item = document.createElement('div');
        item.className = 'conversation-item';
        item.dataset.conversationId = conv.id;

        if (conv.id === currentConversationId) {
            item.classList.add('active');
        }

        const title = document.createElement('div');
        title.className = 'conversation-title';
        title.textContent = conv.title;

        const time = document.createElement('div');
        time.className = 'conversation-time';
        time.textContent = formatTime(conv.updatedAt);

        item.appendChild(title);
        item.appendChild(time);
        conversationsList.appendChild(item);
    });
}

/**
 * Format timestamp for display
 */
function formatTime(isoString) {
    const date = new Date(isoString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;

    return date.toLocaleDateString();
}

// Save messages before page unload
window.addEventListener('beforeunload', () => {
    saveCurrentMessages();
});
