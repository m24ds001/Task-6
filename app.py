import gradio as gr
from transformers import pipeline, MarianMTModel, MarianTokenizer
from langdetect import detect, DetectorFactory
import datetime
import random

# Set seed for consistent language detection
DetectorFactory.seed = 0

class MultilingualChatbot:
    def __init__(self):
        # Initialize sentiment analysis
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        
        # Supported languages
        self.languages = {
            'en': 'English',
            'es': 'Spanish', 
            'fr': 'French',
            'de': 'German',
            'hi': 'Hindi'
        }
        
        # Translation models cache
        self.translation_models = {}
        self.translation_tokenizers = {}
        
        # Cultural greetings and responses
        self.cultural_responses = {
            'en': {
                'greeting': ['Hello!', 'Hi there!', 'Welcome!', 'Greetings!'],
                'farewell': ['Goodbye!', 'See you later!', 'Take care!', 'Bye!'],
                'thanks': ['You\'re welcome!', 'My pleasure!', 'Happy to help!', 'Anytime!'],
                'morning': 'Good morning!',
                'afternoon': 'Good afternoon!',
                'evening': 'Good evening!',
                'help': 'How can I assist you today?'
            },
            'es': {
                'greeting': ['¡Hola!', '¡Bienvenido!', '¡Saludos!'],
                'farewell': ['¡Adiós!', '¡Hasta luego!', '¡Cuídate!'],
                'thanks': ['¡De nada!', '¡Con gusto!', '¡Para servirte!'],
                'morning': '¡Buenos días!',
                'afternoon': '¡Buenas tardes!',
                'evening': '¡Buenas noches!',
                'help': '¿En qué puedo ayudarte hoy?'
            },
            'fr': {
                'greeting': ['Bonjour!', 'Salut!', 'Bienvenue!'],
                'farewell': ['Au revoir!', 'À bientôt!', 'Salut!'],
                'thanks': ['De rien!', 'Avec plaisir!', 'Je vous en prie!'],
                'morning': 'Bonjour!',
                'afternoon': 'Bon après-midi!',
                'evening': 'Bonsoir!',
                'help': 'Comment puis-je vous aider aujourd\'hui?'
            },
            'de': {
                'greeting': ['Hallo!', 'Guten Tag!', 'Willkommen!'],
                'farewell': ['Auf Wiedersehen!', 'Tschüss!', 'Bis bald!'],
                'thanks': ['Gern geschehen!', 'Bitte sehr!', 'Keine Ursache!'],
                'morning': 'Guten Morgen!',
                'afternoon': 'Guten Tag!',
                'evening': 'Guten Abend!',
                'help': 'Wie kann ich Ihnen heute helfen?'
            },
            'hi': {
                'greeting': ['नमस्ते!', 'स्वागत है!'],
                'farewell': ['अलविदा!', 'फिर मिलेंगे!'],
                'thanks': ['स्वागत है!', 'कोई बात नहीं!'],
                'morning': 'सुप्रभात!',
                'afternoon': 'नमस्ते!',
                'evening': 'शुभ संध्या!',
                'help': 'मैं आज आपकी कैसे मदद कर सकता हूं?'
            }
        }
        
        # Intent keywords for different languages
        self.intent_keywords = {
            'greeting': {
                'en': ['hello', 'hi', 'hey', 'greetings'],
                'es': ['hola', 'buenos', 'buenas', 'saludos'],
                'fr': ['bonjour', 'salut', 'bonsoir'],
                'de': ['hallo', 'guten', 'tag', 'morgen'],
                'hi': ['नमस्ते', 'नमस्कार', 'हेलो']
            },
            'farewell': {
                'en': ['bye', 'goodbye', 'see you', 'farewell'],
                'es': ['adiós', 'hasta', 'chao'],
                'fr': ['au revoir', 'adieu', 'salut'],
                'de': ['tschüss', 'auf wiedersehen', 'bis'],
                'hi': ['अलविदा', 'फिर मिलेंगे']
            },
            'thanks': {
                'en': ['thank', 'thanks', 'grateful'],
                'es': ['gracias', 'agradec'],
                'fr': ['merci', 'remercie'],
                'de': ['danke', 'dankeschön'],
                'hi': ['धन्यवाद', 'शुक्रिया']
            }
        }
    
    def detect_language(self, text):
        """Detect the language of input text"""
        try:
            lang = detect(text)
            if lang not in self.languages:
                lang = 'en'
            return lang
        except:
            return 'en'
    
    def get_time_greeting(self, lang):
        """Get culturally appropriate time-based greeting"""
        hour = datetime.datetime.now().hour
        if 5 <= hour < 12:
            return self.cultural_responses[lang]['morning']
        elif 12 <= hour < 17:
            return self.cultural_responses[lang]['afternoon']
        else:
            return self.cultural_responses[lang]['evening']
    
    def detect_intent(self, text, lang):
        """Detect user intent from text"""
        text_lower = text.lower()
        
        for intent, keywords in self.intent_keywords.items():
            if lang in keywords:
                for keyword in keywords[lang]:
                    if keyword in text_lower:
                        return intent
        return None
    
    def get_translation_model(self, src_lang, tgt_lang):
        """Load or retrieve translation model"""
        model_key = f"{src_lang}-{tgt_lang}"
        
        if model_key in self.translation_models:
            return self.translation_models[model_key], self.translation_tokenizers[model_key]
        
        # Model name mapping for Helsinki-NLP models
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            self.translation_models[model_key] = model
            self.translation_tokenizers[model_key] = tokenizer
            return model, tokenizer
        except:
            return None, None
    
    def translate_text(self, text, src_lang, tgt_lang):
        """Translate text between languages"""
        if src_lang == tgt_lang:
            return text
        
        model, tokenizer = self.get_translation_model(src_lang, tgt_lang)
        
        if model is None:
            return text
        
        try:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            translated = model.generate(**inputs)
            return tokenizer.decode(translated[0], skip_special_tokens=True)
        except:
            return text
    
    def generate_response(self, user_input, history, preferred_lang=None):
        """Generate chatbot response with language detection and cultural awareness"""
        
        # Detect language
        detected_lang = self.detect_language(user_input)
        
        # Use preferred language if specified, otherwise use detected
        response_lang = preferred_lang if preferred_lang else detected_lang
        
        # Detect intent
        intent = self.detect_intent(user_input, detected_lang)
        
        # Generate appropriate response based on intent
        if intent == 'greeting':
            time_greeting = self.get_time_greeting(response_lang)
            base_response = f"{time_greeting} {random.choice(self.cultural_responses[response_lang]['greeting'])} {self.cultural_responses[response_lang]['help']}"
        
        elif intent == 'farewell':
            base_response = random.choice(self.cultural_responses[response_lang]['farewell'])
        
        elif intent == 'thanks':
            base_response = random.choice(self.cultural_responses[response_lang]['thanks'])
        
        else:
            # Translate to English for processing if needed
            if detected_lang != 'en':
                english_text = self.translate_text(user_input, detected_lang, 'en')
            else:
                english_text = user_input
            
            # Analyze sentiment
            try:
                sentiment = self.sentiment_analyzer(english_text[:512])[0]
                sentiment_label = sentiment['label']
            except:
                sentiment_label = 'NEUTRAL'
            
            # Generate contextual response in English
            if sentiment_label == 'NEGATIVE':
                english_response = f"I understand you might be feeling frustrated. I'm here to help. Regarding '{english_text[:50]}...', let me assist you with that."
            elif sentiment_label == 'POSITIVE':
                english_response = f"I'm glad to hear from you! About '{english_text[:50]}...', I'd be happy to help with that."
            else:
                english_response = f"Thank you for your message. Regarding '{english_text[:50]}...', I'll do my best to assist you."
            
            # Translate response back to target language
            if response_lang != 'en':
                base_response = self.translate_text(english_response, 'en', response_lang)
            else:
                base_response = english_response
        
        # Add language detection info
        lang_name = self.languages.get(detected_lang, 'Unknown')
        info = f"\n\n_[Detected: {lang_name} | Responding in: {self.languages.get(response_lang, 'Unknown')}]_"
        
        return base_response + info

# Initialize chatbot
chatbot = MultilingualChatbot()

def chat_interface(message, history, language):
    """Interface function for Gradio"""
    lang_code = None if language == "Auto-detect" else language
    response = chatbot.generate_response(message, history, lang_code)
    return response

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🌍 Multilingual AI Chatbot
    
    ### Advanced Features:
    - 🔍 **Automatic Language Detection** (English, Spanish, French, German, Hindi)
    - 🌐 **Seamless Language Switching**
    - 🎭 **Culturally Appropriate Responses**
    - 💭 **Sentiment Analysis**
    - 🕐 **Time-Aware Greetings**
    - 🎯 **Intent Recognition**
    
    Try greeting the bot in different languages or ask questions!
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot_ui = gr.Chatbot(
                height=500,
                label="Chat",
                show_label=True
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type your message in any supported language...",
                    label="Message",
                    scale=4
                )
                language_selector = gr.Dropdown(
                    choices=["Auto-detect", "en", "es", "fr", "de", "hi"],
                    value="Auto-detect",
                    label="Response Language",
                    scale=1
                )
            
            with gr.Row():
                submit = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear")
        
        with gr.Column(scale=1):
            gr.Markdown("""
            ### 📝 Language Codes:
            - **en**: English
            - **es**: Español (Spanish)
            - **fr**: Français (French)
            - **de**: Deutsch (German)
            - **hi**: हिन्दी (Hindi)
            
            ### 💡 Try These:
            - "Hello!"
            - "¡Hola!"
            - "Bonjour!"
            - "Guten Tag!"
            - "नमस्ते!"
            """)
    
    def respond(message, chat_history, language):
        bot_response = chat_interface(message, chat_history, language)
        chat_history.append((message, bot_response))
        return "", chat_history
    
    msg.submit(respond, [msg, chatbot_ui, language_selector], [msg, chatbot_ui])
    submit.click(respond, [msg, chatbot_ui, language_selector], [msg, chatbot_ui])
    clear.click(lambda: None, None, chatbot_ui, queue=False)

if __name__ == "__main__":
    demo.launch()