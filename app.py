from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from datetime import datetime
import os
from supabase import create_client, Client
from typing import Optional
import uuid
from dotenv import load_dotenv
import logging
from configurable_rag import ConfigurableRAG, RAGConfig, SearchMode
import nltk

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')

# Initialize Supabase
supabase: Client = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_KEY')
)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# RAG Configurations
CONFIGS = {
    "hybrid": RAGConfig(
        search_mode=SearchMode.HYBRID,
        initial_k=10,
        final_k=3,
        use_reranking=False,
        bm25_weight=0.3,
        semantic_weight=0.7
    ),
    "hybrid_reranked": RAGConfig(
        search_mode=SearchMode.HYBRID,
        initial_k=10,
        final_k=3,
        use_reranking=True,
        bm25_weight=0.3,
        semantic_weight=0.7
    ),
    "semantic": RAGConfig(
        search_mode=SearchMode.SEMANTIC,
        initial_k=10,
        final_k=3,
        use_reranking=False
    ),
    "semantic_reranked": RAGConfig(
        search_mode=SearchMode.SEMANTIC,
        initial_k=10,
        final_k=3,
        use_reranking=True
    )
}

# Initialize RAG instances
SCHEMA_FILE = "anura_schema.json"
TABLE_DESC_FILE = "table_description.json"
INDEX_DIR = "index"
API_KEY = os.getenv('OPENAI_API_KEY')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/averaged_perceptron_tagger')  # For POS tagging if needed
    nltk.data.find('corpora/stopwords')  # For stopwords if needed
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')

rag_instances = {}
for config_name, config in CONFIGS.items():
    rag_instances[config_name] = ConfigurableRAG(
        SCHEMA_FILE, 
        TABLE_DESC_FILE, 
        API_KEY, 
        config, 
        INDEX_DIR
    )

class User(UserMixin):
    def __init__(self, user_data):
        self.id = user_data['id']
        self.email = user_data['email']

    @staticmethod
    def get(user_id) -> Optional['User']:
        try:
            response = supabase.table('users').select("*").eq('id', user_id).execute()
            if response.data:
                return User(response.data[0])
        except Exception as e:
            logger.error(f"Error fetching user: {e}")
        return None

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

@app.route('/robots933456.txt')
def health_check():
    return ''  # Just return empty response with 200 status code

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            email = request.json.get('email')
            if not email:
                return jsonify({'success': False, 'error': 'Email is required'}), 400

            # Check if user exists
            response = supabase.table('users').select("*").eq('email', email).execute()
            
            if not response.data:
                # Create new user
                user_id = str(uuid.uuid4())
                response = supabase.table('users')\
                    .insert({"id": user_id, "email": email})\
                    .execute()

            user_data = response.data[0]
            user = User(user_data)
            login_user(user)
            return jsonify({'success': True})
            
        except Exception as e:
            logger.error(f"Login error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def home():
    return render_template('index.html', current_user=current_user)

@app.route('/conversations', methods=['GET'])
@login_required
def get_conversations():
    try:
        # First get all conversations
        conv_response = supabase.table('conversations')\
            .select("*")\
            .eq('user_id', current_user.id)\
            .order('created_at', desc=True)\
            .execute()

        conversations = []
        for conv in conv_response.data:
            # For each conversation, get its messages
            msg_response = supabase.table('messages')\
                .select("*, feedback(*)")\
                .eq('conversation_id', conv['id'])\
                .order('created_at')\
                .execute()

            messages = []
            for msg in msg_response.data:
                feedback = None
                if msg.get('feedback') and msg['feedback']:
                    feedback = {
                        'is_positive': msg['feedback'][0]['is_positive'] if msg['feedback'] else None,
                        'comment': msg['feedback'][0]['comment'] if msg['feedback'] else None
                    }
                
                messages.append({
                    'id': msg['id'],
                    'content': msg['content'],
                    'is_user': msg['is_user'],
                    'feedback': feedback
                })

            conversations.append({
                'id': conv['id'],
                'title': conv['title'],
                'messages': messages
            })

        return jsonify(conversations)
    except Exception as e:
        logger.error(f"Error fetching conversations: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/conversations', methods=['POST'])
@login_required
def create_conversation():
    try:
        data = request.json
        conversation_id = str(uuid.uuid4())
        conversation_data = {
            'id': conversation_id,
            'title': data.get('title', 'New Chat'),
            'user_id': current_user.id
        }
        
        response = supabase.table('conversations').insert(conversation_data).execute()
        new_conversation = response.data[0]
        
        return jsonify({
            'id': new_conversation['id'],
            'title': new_conversation['title'],
            'messages': []
        })
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/conversations/<conversation_id>', methods=['DELETE'])
@login_required
def delete_conversation(conversation_id):
    try:
        # Verify ownership before deletion
        response = supabase.table('conversations')\
            .delete()\
            .eq('id', conversation_id)\
            .eq('user_id', current_user.id)\
            .execute()
            
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/conversations/<conversation_id>/messages', methods=['GET'])
@login_required
def get_conversation_messages(conversation_id):
    try:
        # Verify conversation belongs to user
        conv_response = supabase.table('conversations')\
            .select("*")\
            .eq('id', conversation_id)\
            .eq('user_id', current_user.id)\
            .execute()
            
        if not conv_response.data:
            return jsonify({'error': 'Conversation not found'}), 404
        
        # Get messages with feedback
        msg_response = supabase.table('messages')\
            .select("*, feedback(*)")\
            .eq('conversation_id', conversation_id)\
            .order('created_at')\
            .execute()
            
        messages = []
        for msg in msg_response.data:
            feedback = None
            if msg.get('feedback') and msg['feedback']:
                feedback = {
                    'is_positive': msg['feedback'][0]['is_positive'] if msg['feedback'] else None,
                    'comment': msg['feedback'][0]['comment'] if msg['feedback'] else None
                }
            
            messages.append({
                'id': msg['id'],
                'content': msg['content'],
                'is_user': msg['is_user'],
                'feedback': feedback
            })
            
        return jsonify(messages)
        
    except Exception as e:
        logger.error(f"Error fetching conversation messages: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/send_message', methods=['POST'])
@login_required
def send_message():
    try:
        data = request.json
        message = data.get('message')
        conversation_id = data.get('conversation_id')
        use_memory = data.get('use_memory', False)
        rag_type = data.get('rag_type', 'semantic')
        
        # Verify conversation belongs to user
        conv_response = supabase.table('conversations')\
            .select("*")\
            .eq('id', conversation_id)\
            .eq('user_id', current_user.id)\
            .execute()
            
        if not conv_response.data:
            return jsonify({'error': 'Conversation not found'}), 404
        
        # Get conversation history if memory is enabled
        chat_history = []
        if use_memory:
            messages_response = supabase.table('messages')\
                .select("*")\
                .eq('conversation_id', conversation_id)\
                .order('created_at')\
                .execute()
                
            for msg in messages_response.data:
                role = "user" if msg['is_user'] else "assistant"
                chat_history.append({"role": role, "content": msg['content']})
        
        # Save user message
        user_message_id = str(uuid.uuid4())
        user_message_data = {
            'id': user_message_id,
            'content': message,
            'is_user': True,
            'conversation_id': conversation_id
        }
        user_msg_response = supabase.table('messages').insert(user_message_data).execute()
        
        # Get appropriate RAG instance and generate response
        rag = rag_instances.get(rag_type, rag_instances['semantic'])
        
        try:
            if use_memory and chat_history:
                full_context = "\n\nChat History:\n"
                for msg in chat_history[-5:]:
                    speaker = "User" if msg["role"] == "user" else "Assistant"
                    full_context += f"{speaker}: {msg['content']}\n"
                full_context += f"\nCurrent Question: {message}"
                query = full_context
            else:
                query = message
                
            response, retrieved_docs_with_scores, execution_time = rag.query(query)
            
            # Save bot response
            bot_message_id = str(uuid.uuid4())
            bot_message_data = {
                'id': bot_message_id,
                'content': response,
                'is_user': False,
                'conversation_id': conversation_id
            }
            bot_msg_response = supabase.table('messages').insert(bot_message_data).execute()
            
            # Update conversation title if it's the first message
            if len(chat_history) <= 2:
                new_title = message[:30] + ('...' if len(message) > 30 else '')
                supabase.table('conversations')\
                    .update({'title': new_title})\
                    .eq('id', conversation_id)\
                    .execute()
            
            context = {
                "sources": [
                    {
                        "title": "Table Info" if doc.metadata.get('type') == 'table' else "Column Info",
                        "content": doc.content,
                        "relevance": float(score),
                        "type": doc.metadata.get('type', 'unknown')
                    }
                    for doc, score in retrieved_docs_with_scores
                ]
            }
            
            return jsonify({
                "response": response,
                "message_id": bot_message_id,
                "context": context,
                "updatedTitle": new_title if len(chat_history) <= 2 else None
            })
            
        except Exception as e:
            logger.error(f"RAG Error: {str(e)}")
            error_response = "I apologize, but I encountered an error processing your query. Please try again."
            
            # Save error response
            error_message_id = str(uuid.uuid4())
            error_message_data = {
                'id': error_message_id,
                'content': error_response,
                'is_user': False,
                'conversation_id': conversation_id
            }
            error_msg_response = supabase.table('messages').insert(error_message_data).execute()
            
            return jsonify({
                "response": error_response,
                "message_id": error_message_id,
                "context": {
                    "sources": [
                        {
                            "title": "Error",
                            "content": "Failed to retrieve context due to an error.",
                            "relevance": 0.0,
                            "type": "error"
                        }
                    ]
                }
            })
            
    except Exception as e:
        logger.error(f"Error in send_message: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
@login_required
def submit_feedback():
    try:
        data = request.json
        message_id = data.get('message_id')
        is_positive = data.get('is_positive')
        comment = data.get('comment')

        # Verify message belongs to user's conversation
        msg_response = supabase.table('messages')\
            .select("*, conversations!inner(*)")\
            .eq('id', message_id)\
            .eq('conversations.user_id', current_user.id)\
            .execute()
            
        if not msg_response.data:
            return jsonify({'error': 'Message not found'}), 404

        # Delete existing feedback if it exists
        supabase.table('feedback')\
            .delete()\
            .eq('message_id', message_id)\
            .execute()

        # Create new feedback
        feedback_id = str(uuid.uuid4())
        feedback_data = {
            'id': feedback_id,
            'message_id': message_id,
            'is_positive': is_positive,
            'comment': comment
        }
        supabase.table('feedback').insert(feedback_data).execute()

        return jsonify({'success': True})
    
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(Exception)
def handle_error(error):
    logger.error(f"Unhandled error: {error}", exc_info=True)
    return jsonify({
        'error': 'An unexpected error occurred',
        'message': str(error) if app.debug else 'Please try again later'
    }), 500

if __name__ == '__main__':
    app.run(debug=False)