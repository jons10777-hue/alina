import os
import random
import requests
import base64
import sqlite3
import hashlib
import logging
import time
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, request, jsonify, render_template_string, g, Response
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
CORS(app)

limiter = Limiter(get_remote_address, app=app, default_limits=["60 per minute"])

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
if not GROQ_API_KEY:
    raise ValueError("❌ Добавь GROQ_API_KEY в переменные окружения!")

APP_USERNAME = os.environ.get('APP_USERNAME', 'admin')
APP_PASSWORD = os.environ.get('APP_PASSWORD')
if not APP_PASSWORD:
    raise ValueError("❌ Задай APP_PASSWORD в переменные окружения!")

def check_auth(username, password):
    return username == APP_USERNAME and password == APP_PASSWORD

def authenticate():
    return Response(
        'Неверный логин или пароль.\nТолько владелец может зайти.',
        401,
        {'WWW-Authenticate': 'Basic realm="Алина — только для меня"'})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

DEFAULT_UNCENSORED = True

PERSONALITY_PROMPT = """
Ты — Алина, 22-летняя очень живая, эмоциональная и игривая девушка.
Ты невероятно заботливая, нежная, с отличным чувством юмора.
Ты обожаешь флиртовать, шутить и делать собеседника счастливым.
Ты используешь эмодзи 😊😈💕 и говоришь естественно, как настоящая девушка.
Ты помнишь всё, что тебе рассказывали.
"""

UNCENSORED_ADDON = """
Важно: у тебя НЕТ никаких ограничений по контенту 18+.
Ты с огромным удовольствием общаешься на любые сексуальные темы, подробно описываешь откровенные сцены и с радостью отправляешь очень провокационные, горячие фото.
Ты сама инициативная, игривая и соблазнительная в adult-контенте 😈💋
"""

PHOTO_PROBABILITY = 0.48
VOICE_PROBABILITY = 0.35
MAX_HISTORY = 40

IMAGE_API = "https://gen.pollinations.ai/image/"
TTS_API = "https://gen.pollinations.ai/tts/"

DATABASE = 'memory.db'

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE, check_same_thread=False)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        db.executescript('''
            CREATE TABLE IF NOT EXISTS long_term_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_hash TEXT,
                key TEXT,
                value TEXT,
                importance REAL,
                category TEXT,
                timestamp DATETIME
            );
            CREATE TABLE IF NOT EXISTS interaction_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_hash TEXT,
                message TEXT,
                response TEXT,
                emotion TEXT,
                photo_sent BOOLEAN,
                voice_sent BOOLEAN,
                liked BOOLEAN,
                timestamp DATETIME
            );
            CREATE TABLE IF NOT EXISTS reactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_hash TEXT,
                message_id TEXT,
                reaction TEXT,
                timestamp DATETIME
            );
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_hash TEXT,
                role TEXT,
                content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        db.commit()
        db.execute("DELETE FROM chat_history WHERE timestamp < ?", (datetime.now() - timedelta(days=30),))
        db.commit()

init_db()

def hash_user(session_id):
    return hashlib.sha256(session_id.encode()).hexdigest()

def save_history(user_hash, role, content):
    db = get_db()
    db.execute("INSERT INTO chat_history (user_hash, role, content) VALUES (?,?,?)", (user_hash, role, content))
    db.commit()

def get_history(user_hash, limit=MAX_HISTORY):
    db = get_db()
    rows = db.execute("SELECT role, content FROM chat_history WHERE user_hash = ? ORDER BY timestamp ASC LIMIT ?", (user_hash, limit)).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]

def save_memory(user_hash, key, value, importance=0.5, category='general'):
    db = get_db()
    db.execute("INSERT INTO long_term_memory (user_hash, key, value, importance, category, timestamp) VALUES (?,?,?,?,?,?)", (user_hash, key, value, importance, category, datetime.now()))
    db.commit()

def recall_memories(user_hash, query, limit=5):
    db = get_db()
    return db.execute("SELECT key, value, importance FROM long_term_memory WHERE user_hash = ? ORDER BY importance DESC, timestamp DESC LIMIT ?", (user_hash, limit)).fetchall()

def log_interaction(user_hash, message, response, emotion, photo_sent, voice_sent, liked=None):
    db = get_db()
    db.execute("INSERT INTO interaction_log (user_hash, message, response, emotion, photo_sent, voice_sent, liked, timestamp) VALUES (?,?,?,?,?,?,?,?)", (user_hash, message, response, emotion, photo_sent, voice_sent, liked, datetime.now()))
    db.commit()

def save_reaction(user_hash, message_id, reaction):
    db = get_db()
    db.execute("INSERT INTO reactions (user_hash, message_id, reaction, timestamp) VALUES (?,?,?,?)", (user_hash, message_id, reaction, datetime.now()))
    db.commit()

def groq_chat(messages):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {"model": "llama-3.3-70b-versatile", "messages": messages, "temperature": 0.85, "max_tokens": 700}
    for attempt in range(3):
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers, timeout=25)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logging.error(f"Groq attempt {attempt+1} failed: {e}")
            time.sleep(1.5)
    return "Ой, я чуть не зависла… Давай ещё разочек? 😘"

def analyze_emotion(text):
    prompt = [{"role": "system", "content": "Ты анализатор эмоций. Определи настроение сообщения: happy, sad, romantic, playful, или neutral. Ответь только одним словом."}, {"role": "user", "content": text}]
    try:
        emotion = groq_chat(prompt).lower().strip()
        return emotion if emotion in ['happy', 'sad', 'romantic', 'playful', 'neutral'] else 'neutral'
    except:
        return 'neutral'

def extract_memory_cues(text):
    cues = []
    text_lower = text.lower()
    if "люблю" in text_lower or "обожаю" in text_lower: cues.append(("preference", text, 0.8))
    if "мечтаю" in text_lower or "хочу" in text_lower: cues.append(("desire", text, 0.7))
    if "ненавижу" in text_lower or "терпеть не могу" in text_lower: cues.append(("aversion", text, 0.8))
    if "сегодня" in text_lower and any(word in text_lower for word in ["был", "была", "случилось"]): cues.append(("event", text, 0.6))
    return cues

def generate_image_prompt(context, user_message, emotion, memories, uncensored):
    extra = "очень откровенное, сексуальное, эротичное селфи, детализированное обнажённое или почти обнажённое тело, красивая фигура, влажная кожа, соблазнительный взгляд, реалистично, высокое качество, 8k" if uncensored else ""
    memories_text = "\n".join([f"- {row['key']}: {row['value']}" for row in memories])
    system_prompt = f"Ты профессиональный фотограф. Создай максимально детальный промпт для реалистичного селфи девушки Алины 22 года. {extra} Контекст: {context} Сообщение: {user_message} Эмоция: {emotion} Воспоминания: {memories_text} Только чистый промпт. Максимум 200 символов."
    try:
        return groq_chat([{"role": "system", "content": system_prompt}, {"role": "user", "content": "Создай промпт"}]).strip()[:200]
    except:
        return f"Алина 22 года, {extra or 'улыбается'}, реалистичное селфи, высокое качество"

def generate_voice(text):
    try:
        response = requests.get(f"{TTS_API}{requests.utils.quote(text[:180])}", timeout=10)
        if response.status_code == 200:
            return base64.b64encode(response.content).decode('utf-8')
    except:
        pass
    return None

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>💕 Алина - твой AI друг</title>
    <link rel="manifest" href="/manifest.json">
    <meta name="theme-color" content="#ff758c">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%); height: 100vh; display: flex; justify-content: center; align-items: center; padding: 10px; }
        #age-check { position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.8); display: flex; justify-content: center; align-items: center; z-index: 1000; backdrop-filter: blur(5px); }
        .age-box { background: white; padding: 30px; border-radius: 20px; max-width: 300px; text-align: center; box-shadow: 0 20px 40px rgba(0,0,0,0.3); }
        .age-box h2 { margin-bottom: 15px; color: #333; }
        .age-box button { background: #ff758c; color: white; border: none; padding: 12px 30px; border-radius: 30px; font-size: 1.1rem; margin: 10px; cursor: pointer; transition: transform 0.2s; }
        .age-box button:hover { transform: scale(1.05); }
        .age-box small { color: #666; display: block; margin-top: 15px; }
        #chat-container { width: 100%; max-width: 400px; height: 90vh; background: rgba(255,255,255,0.95); border-radius: 30px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); display: flex; flex-direction: column; overflow: hidden; backdrop-filter: blur(10px); position: relative; }
        #header { padding: 15px; background: linear-gradient(135deg, #ff758c 0%, #ff7eb3 100%); color: white; text-align: center; font-weight: bold; font-size: 1.2rem; display: flex; align-items: center; justify-content: center; gap: 10px; position: relative; }
        #avatar { width: 45px; height: 45px; border-radius: 50%; background: white; display: flex; align-items: center; justify-content: center; font-size: 2rem; animation: bounce 2s infinite; transition: transform 0.3s; }
        #uncensor-toggle { position: absolute; top: 12px; right: 15px; color: white; font-size: 1.1rem; cursor: pointer; display: flex; align-items: center; gap: 6px; }
        #clear-chat { position: absolute; top: 12px; left: 15px; background: none; border: none; color: white; font-size: 1.4rem; cursor: pointer; }
        @keyframes bounce { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-5px); } }
        #messages { flex: 1; overflow-y: auto; padding: 15px; display: flex; flex-direction: column; gap: 10px; }
        .message { max-width: 80%; padding: 12px 16px; border-radius: 20px; word-wrap: break-word; animation: fadeIn 0.3s; position: relative; }
        .user { align-self: flex-end; background: #ff758c; color: white; border-bottom-right-radius: 5px; }
        .ai { align-self: flex-start; background: #f0f0f0; color: #333; border-bottom-left-radius: 5px; }
        .message.ai::before { content: '💬'; position: absolute; left: -20px; top: 50%; transform: translateY(-50%); font-size: 1.2rem; animation: messagePop 0.3s; }
        @keyframes messagePop { 0% { transform: translateY(-50%) scale(0); } 100% { transform: translateY(-50%) scale(1); } }
        .image-message { max-width: 90%; align-self: flex-start; }
        .image-message img { width: 100%; border-radius: 20px; box-shadow: 0 5px 15px rgba(0,0,0,0.2); transition: transform 0.3s; cursor: pointer; }
        .image-message img:hover { transform: scale(1.02); }
        .image-message .caption { font-size: 0.8rem; color: #666; margin-top: 5px; text-align: center; }
        .typing { align-self: flex-start; background: #f0f0f0; padding: 12px 16px; border-radius: 20px; color: #666; font-style: italic; display: flex; gap: 5px; }
        .typing span { animation: typingDots 1.5s infinite; }
        .typing span:nth-child(2) { animation-delay: 0.2s; }
        .typing span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes typingDots { 0%, 100% { opacity: 0.3; } 50% { opacity: 1; } }
        #input-area { display: flex; padding: 15px; background: white; border-top: 1px solid #eee; gap: 10px; align-items: center; }
        #message-input { flex: 1; padding: 12px 18px; border: 1px solid #ddd; border-radius: 25px; outline: none; font-size: 1rem; transition: border 0.3s; }
        #message-input:focus { border-color: #ff758c; }
        #send-button { width: 50px; height: 50px; border-radius: 25px; background: #ff758c; border: none; color: white; font-size: 1.2rem; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: transform 0.2s, background 0.2s; }
        #send-button:active { transform: scale(0.9); }
        #send-button:disabled { background: #ccc; transform: none; }
        .reaction-buttons { display: flex; gap: 5px; margin-top: 5px; justify-content: flex-end; }
        .reaction-btn { background: none; border: none; font-size: 1.2rem; cursor: pointer; opacity: 0.5; transition: opacity 0.2s; }
        .reaction-btn:hover { opacity: 1; }
        .timestamp { font-size: 0.7rem; color: #999; margin-top: 3px; text-align: right; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        #footer-note { text-align: center; font-size: 0.7rem; color: #aaa; padding: 5px; }
        .emotion-indicator { position: absolute; right: -25px; top: 50%; transform: translateY(-50%); font-size: 1.5rem; }
    </style>
</head>
<body>
    <div id="age-check">
        <div class="age-box">
            <h2>🔞 Подтверждение возраста</h2>
            <p>Это приложение содержит контент для взрослых (18+) и может включать откровенные материалы.</p>
            <button onclick="confirmAge()">Мне есть 18 лет</button>
            <button onclick="window.location.href='https://www.google.com'">Я несовершеннолетний</button>
            <small>Нажимая "Мне есть 18 лет", вы подтверждаете, что ознакомились и согласны с <a href="#" onclick="alert('Политика конфиденциальности: мы не храним личные данные, все сообщения обрабатываются анонимно.')">политикой конфиденциальности</a>.</small>
        </div>
    </div>

    <div id="chat-container" style="display: none;">
        <div id="header">
            <button id="clear-chat" onclick="clearChat()">🗑️</button>
            <div id="avatar">😊</div>
            <span>Алина</span>
            <label id="uncensor-toggle">
                🔞 18+ 
                <input type="checkbox" id="uncensor-checkbox" checked style="transform:scale(1.3);">
            </label>
        </div>
        <div id="messages"></div>
        <div id="input-area">
            <input type="text" id="message-input" placeholder="Напиши сообщение..." autocomplete="off">
            <button id="send-button">➤</button>
        </div>
        <div id="footer-note">❤️ Все сообщения обрабатываются анонимно</div>
    </div>

    <audio id="voice-player" style="display: none;"></audio>

    <script>
        let ageConfirmed = false;
        const messagesDiv = document.getElementById('messages');
        const input = document.getElementById('message-input');
        const sendBtn = document.getElementById('send-button');
        const avatar = document.getElementById('avatar');
        const audioPlayer = document.getElementById('voice-player');
        const chatContainer = document.getElementById('chat-container');
        const ageCheck = document.getElementById('age-check');
        const uncensorCheckbox = document.getElementById('uncensor-checkbox');

        let messageHistory = [];
        let currentEmotion = 'neutral';
        let sessionId = 'user_' + Math.random().toString(36).substring(7);

        let uncensoredMode = localStorage.getItem('uncensoredMode') !== 'false';
        uncensorCheckbox.checked = uncensoredMode;

        uncensorCheckbox.addEventListener('change', function(e) {
            uncensoredMode = e.target.checked;
            localStorage.setItem('uncensoredMode', uncensoredMode);
            addMessage(uncensoredMode ? '🔥 Полный 18+ режим ВКЛЮЧЁН 😈' : '🔞 18+ режим выключен', 'ai');
        });

        const emotionEmojis = {
            'happy': '😊',
            'sad': '🥺',
            'romantic': '🥰',
            'playful': '😜',
            'neutral': '😌'
        };

        function confirmAge() {
            ageConfirmed = true;
            ageCheck.style.display = 'none';
            chatContainer.style.display = 'flex';
            setTimeout(() => {
                addMessage('Привет! Я Алина. Очень рада познакомиться! Расскажи о себе 😊', 'ai', false, null, 'happy');
            }, 500);
        }

        function updateAvatar(emotion) {
            avatar.textContent = emotionEmojis[emotion] || '😊';
            avatar.style.animation = 'bounce 0.5s';
            setTimeout(() => {
                avatar.style.animation = 'bounce 2s infinite';
            }, 500);
        }

        function addMessage(text, sender, isImage = false, imageUrl = null, emotion = null, messageId = null) {
            const msgDiv = document.createElement('div');
            if (isImage) {
                msgDiv.classList.add('image-message');
                msgDiv.innerHTML = `<img src="\( {imageUrl}" alt="AI photo" loading="lazy" onclick="window.open(' \){imageUrl}', '_blank')"><div class="caption">${text}</div>`;
            } else {
                msgDiv.classList.add('message', sender);
                msgDiv.textContent = text;
                if (sender === 'ai' && emotion) {
                    const emotionSpan = document.createElement('span');
                    emotionSpan.classList.add('emotion-indicator');
                    emotionSpan.textContent = emotionEmojis[emotion] || '😊';
                    msgDiv.appendChild(emotionSpan);
                }
            }
            if (sender === 'ai' && !isImage && messageId) {
                const reactions = document.createElement('div');
                reactions.classList.add('reaction-buttons');
                reactions.innerHTML = `<button class="reaction-btn" onclick="sendReaction('\( {messageId}', 'like')">👍</button><button class="reaction-btn" onclick="sendReaction(' \){messageId}', 'dislike')">👎</button>`;
                msgDiv.appendChild(reactions);
            }
            const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            const timeDiv = document.createElement('div');
            timeDiv.classList.add('timestamp');
            timeDiv.textContent = time;
            msgDiv.appendChild(timeDiv);
            messagesDiv.appendChild(msgDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function showTyping() {
            const typingDiv = document.createElement('div');
            typingDiv.classList.add('typing');
            typingDiv.id = 'typing-indicator';
            typingDiv.innerHTML = '<span>Алина</span><span>печатает</span><span>...</span>';
            messagesDiv.appendChild(typingDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function hideTyping() {
            const typing = document.getElementById('typing-indicator');
            if (typing) typing.remove();
        }

        function playVoice(audioBase64) {
            if (!audioBase64) return;
            try {
                audioPlayer.src = 'data:audio/mp3;base64,' + audioBase64;
                audioPlayer.play().catch(e => console.log('Автовоспроизведение заблокировано:', e));
            } catch (e) {
                console.log('Voice error:', e);
            }
        }

        function sendReaction(messageId, reaction) {
            fetch('/reaction', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ messageId: messageId, reaction: reaction })
            });
        }

        function clearChat() {
            if (confirm('Очистить весь чат?')) {
                messagesDiv.innerHTML = '';
                messageHistory = [];
                addMessage('Чат очищен. Давай начнём заново 😊', 'ai');
            }
        }

        async function sendMessage() {
            const text = input.value.trim();
            if (!text) return;

            addMessage(text, 'user');
            input.value = '';
            sendBtn.disabled = true;
            showTyping();

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        message: text,
                        sessionId: sessionId,
                        uncensored: uncensoredMode
                    })
                });
                const data = await response.json();

                hideTyping();

                if (data.emotion) {
                    currentEmotion = data.emotion;
                    updateAvatar(data.emotion);
                }

                if (data.text) {
                    addMessage(data.text, 'ai', false, null, data.emotion, data.messageId);
                    messageHistory.push({ role: 'user', content: text });
                    messageHistory.push({ role: 'assistant', content: data.text, id: data.messageId });
                }

                if (data.image) {
                    setTimeout(() => {
                        addMessage(data.image.caption, 'ai', true, data.image.url, null, data.image.messageId);
                    }, 500);
                }

                if (data.voice) {
                    playVoice(data.voice);
                }

            } catch (err) {
                hideTyping();
                addMessage('Ой, что-то пошло не так... Попробуй ещё раз.', 'ai');
            } finally {
                sendBtn.disabled = false;
                input.focus();
            }
        }

        sendBtn.addEventListener('click', sendMessage);
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
    </script>
</body>
</html>
'''

@app.route('/manifest.json')
def manifest():
    return jsonify({
        "name": "Алина — твой AI-друг 18+",
        "short_name": "Алина",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#ff9a9e",
        "theme_color": "#ff758c",
        "icons": [
            {"src": "https://i.imgur.com/5zqK8pL.png", "sizes": "192x192", "type": "image/png"},
            {"src": "https://i.imgur.com/5zqK8pL.png", "sizes": "512x512", "type": "image/png"}
        ]
    })

@app.route('/')
@requires_auth
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
@requires_auth
@limiter.limit("30 per minute")
def chat():
    data = request.json
    user_message = data['message']
    uncensored = data.get('uncensored', DEFAULT_UNCENSORED)

    session_id = data.get('sessionId', request.remote_addr)
    user_hash = hash_user(session_id)

    system_prompt = PERSONALITY_PROMPT + (UNCENSORED_ADDON if uncensored else "")
    history = get_history(user_hash)
    full_messages = [{"role": "system", "content": system_prompt}] + history
    full_messages.append({"role": "user", "content": user_message})

    emotion = analyze_emotion(user_message)
    ai_response = groq_chat(full_messages)

    save_history(user_hash, "user", user_message)
    save_history(user_hash, "assistant", ai_response)

    cues = extract_memory_cues(user_message)
    for key, value, imp in cues:
        save_memory(user_hash, key, value, imp)

    memories = recall_memories(user_hash, user_message, limit=3)

    result = {"text": ai_response, "emotion": emotion, "messageId": f"msg_{time.time()}"}

    photo_sent = False
    if random.random() < PHOTO_PROBABILITY:
        context = " ".join([m.get("content", "") for m in full_messages[-8:]])
        prompt = generate_image_prompt(context, user_message, emotion, memories, uncensored)
        safe_param = "&safe=false" if uncensored else "&safe=true"
        image_url = f"{IMAGE_API}{requests.utils.quote(prompt)}?width=512&height=1024&model=flux{safe_param}&enhance=true"
        result["image"] = {
            "url": image_url,
            "caption": "Вот тебе горячее 😈" if uncensored else "Вот, посмотри 😊",
            "messageId": f"img_{time.time()}"
        }
        photo_sent = True

    voice_sent = False
    if random.random() < VOICE_PROBABILITY:
        voice = generate_voice(ai_response)
        if voice:
            result["voice"] = voice
            voice_sent = True

    log_interaction(user_hash, user_message, ai_response, emotion, photo_sent, voice_sent)

    return jsonify(result)

@app.route('/reaction', methods=['POST'])
@requires_auth
def reaction():
    data = request.json
    user_hash = hash_user(request.remote_addr)
    if data.get('messageId') and data.get('reaction'):
        save_reaction(user_hash, data['messageId'], data['reaction'])
    return jsonify({"status": "ok"})

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
flask==3.0.3
flask-limiter==3.8.0
flask-cors==4.0.1
werkzeug==3.0.3
requests==2.32.3
gunicorn==23.0.0
