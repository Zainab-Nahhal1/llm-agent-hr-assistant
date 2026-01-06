"""
AI-Powered HR Assistant with Tool Calling
Built with LangChain, Flask, and Modern Web Interface
"""

import os
import json
from typing import Dict, List, Any, Optional
from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent

# Mock Data and Tools

EMPLOYEE_DB = {
    "E12345": {"name": "John Doe", "department": "Engineering", "role": "Senior Software Engineer", "email": "john.doe@company.com", "join_date": "2020-03-15"},
    "E12346": {"name": "Jane Smith", "department": "Data Science", "role": "Data Scientist", "email": "jane.smith@company.com", "join_date": "2021-06-20"},
    "E12347": {"name": "Mike Johnson", "department": "HR", "role": "HR Manager", "email": "mike.johnson@company.com", "join_date": "2019-01-10"}
}

LEAVE_BALANCE_DB = {
    "E12345": {"remaining_leave_days": 12, "total_leave_days": 20, "used_leave_days": 8},
    "E12346": {"remaining_leave_days": 18, "total_leave_days": 20, "used_leave_days": 2},
    "E12347": {"remaining_leave_days": 5, "total_leave_days": 20, "used_leave_days": 15}
}

INTERVIEW_QUESTIONS = {
    "Software Engineer": ["What is Object-Oriented Programming and its main principles?", "Explain the difference between REST and GraphQL.", "What are design patterns? Give examples.", "How do you handle version control in a team?", "Describe your approach to debugging complex issues."],
    "Data Scientist": ["What is the difference between supervised and unsupervised learning?", "Explain the bias-variance tradeoff.", "How do you handle missing data in a dataset?", "What is cross-validation and why is it important?", "Describe a machine learning project you've worked on."],
    "HR Manager": ["How do you handle conflict resolution in the workplace?", "What strategies do you use for employee retention?", "How do you ensure fair and unbiased hiring?", "Describe your experience with performance management.", "How do you stay updated with labor laws and regulations?"],
    "Product Manager": ["How do you prioritize features in a product roadmap?", "What is your approach to gathering user feedback?", "How do you work with engineering and design teams?", "Describe a time you had to pivot a product strategy.", "How do you measure product success?"]
}

COMPANY_POLICIES = {
    "remote_work": {"policy": "Employees can work remotely up to 3 days per week after completing 6 months with the company.", "approval_required": True, "eligible_departments": ["Engineering", "Data Science", "Marketing"]},
    "salary": {"policy": "Salaries are reviewed annually in Q1. Performance-based raises range from 3-15%.", "bonus_structure": "Annual bonus eligibility based on company and individual performance."},
    "leave": {"policy": "20 days annual leave, 10 days sick leave, 5 days personal leave per year.", "rollover": "Up to 5 unused leave days can be rolled over to the next year."},
    "working_hours": {"policy": "Standard working hours are 9 AM - 6 PM, Monday to Friday.", "flexibility": "Flexible hours available with manager approval."}
}

@tool
def get_employee_details(employee_id: str) -> str:
    """Retrieve employee details from the database."""
    if employee_id in EMPLOYEE_DB:
        return json.dumps({"success": True, "data": EMPLOYEE_DB[employee_id]})
    return json.dumps({"success": False, "error": f"Employee ID {employee_id} not found."})

@tool
def check_leave_balance(employee_id: str) -> str:
    """Check the leave balance for an employee."""
    if employee_id in LEAVE_BALANCE_DB:
        return json.dumps({"success": True, "data": LEAVE_BALANCE_DB[employee_id]})
    return json.dumps({"success": False, "error": f"Leave balance not found for {employee_id}."})

@tool
def generate_interview_questions(job_role: str) -> str:
    """Generate interview questions for a specific job role."""
    normalized_role = job_role.strip().title()
    if normalized_role in INTERVIEW_QUESTIONS:
        return json.dumps({"success": True, "job_role": normalized_role, "questions": INTERVIEW_QUESTIONS[normalized_role]})
    return json.dumps({"success": True, "job_role": job_role, "questions": ["Tell me about yourself and your experience.", "Why are you interested in this position?", "What are your greatest strengths and weaknesses?", "Describe a challenging situation and how you handled it.", "Where do you see yourself in 5 years?"], "note": f"Generic questions provided for '{job_role}'."})

@tool
def get_company_policy(policy_type: str) -> str:
    """Retrieve company policy information."""
    normalized_type = policy_type.lower().replace(" ", "_")
    if normalized_type in COMPANY_POLICIES:
        return json.dumps({"success": True, "policy_type": policy_type, "data": COMPANY_POLICIES[normalized_type]})
    return json.dumps({"success": False, "error": f"Policy '{policy_type}' not found."})

# Setup Agent

def setup_hr_assistant(openai_api_key: str):
    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.7, openai_api_key=openai_api_key, streaming=True)
    tools = [get_employee_details, check_leave_balance, generate_interview_questions, get_company_policy]
    system_message = """You are a helpful HR assistant. Help employees with HR policies, benefits, procedures, employee information, leave balances, interview questions, and company policies. Be friendly, professional, and helpful."""
    return create_react_agent(llm, tools, prompt=system_message)

# Flask App

app = Flask(__name__)
CORS(app)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "<YOUR_OPENAI_API_KEY>")
agent_executor = setup_hr_assistant(OPENAI_API_KEY)
chat_sessions = {}

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI HR Assistant</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; display: flex; justify-content: center; align-items: center; padding: 20px; }
        .container { width: 100%; max-width: 900px; background: white; border-radius: 20px; box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3); overflow: hidden; display: flex; flex-direction: column; height: 90vh; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; }
        .header h1 { font-size: 28px; margin-bottom: 10px; }
        .header p { font-size: 14px; opacity: 0.9; }
        .quick-actions { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; margin-top: 15px; }
        .quick-btn { background: rgba(255, 255, 255, 0.2); border: 1px solid rgba(255, 255, 255, 0.3); color: white; padding: 8px 16px; border-radius: 20px; font-size: 12px; cursor: pointer; transition: all 0.3s; }
        .quick-btn:hover { background: rgba(255, 255, 255, 0.3); transform: translateY(-2px); }
        .chat-container { flex: 1; overflow-y: auto; padding: 30px; background: #f8f9fa; }
        .message { margin-bottom: 20px; animation: slideIn 0.3s ease-out; }
        @keyframes slideIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .message.user { text-align: right; }
        .message-content { display: inline-block; max-width: 70%; padding: 15px 20px; border-radius: 18px; word-wrap: break-word; }
        .message.user .message-content { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-bottom-right-radius: 4px; }
        .message.assistant .message-content { background: white; color: #333; border-bottom-left-radius: 4px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1); }
        .input-container { padding: 20px 30px; background: white; border-top: 1px solid #e0e0e0; display: flex; gap: 10px; }
        #userInput { flex: 1; padding: 15px 20px; border: 2px solid #e0e0e0; border-radius: 25px; font-size: 15px; outline: none; transition: border 0.3s; }
        #userInput:focus { border-color: #667eea; }
        .btn { padding: 15px 30px; border: none; border-radius: 25px; font-size: 15px; cursor: pointer; transition: all 0.3s; font-weight: 600; }
        .btn-primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
        .btn-primary:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4); }
        .btn-primary:disabled { opacity: 0.6; cursor: not-allowed; }
        .btn-secondary { background: #f0f0f0; color: #666; }
        .btn-secondary:hover { background: #e0e0e0; }
        .typing-indicator { display: none; padding: 15px 20px; background: white; border-radius: 18px; width: fit-content; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1); }
        .typing-indicator span { height: 10px; width: 10px; background: #667eea; border-radius: 50%; display: inline-block; margin: 0 3px; animation: bounce 1.4s infinite ease-in-out both; }
        .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
        .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
        @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }
        .info-badge { display: inline-block; background: #e3f2fd; color: #1976d2; padding: 4px 12px; border-radius: 12px; font-size: 12px; margin: 5px 5px 5px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏢 AI HR Assistant</h1>
            <p>Your intelligent HR companion powered by AI</p>
            <div class="quick-actions">
                <button class="quick-btn" id="q1">📅 Leave Balance</button>
                <button class="quick-btn" id="q2">👤 Employee Info</button>
                <button class="quick-btn" id="q3">💼 Interview Questions</button>
                <button class="quick-btn" id="q4">🏠 Policies</button>
            </div>
        </div>
        <div class="chat-container" id="chatContainer">
            <div class="message assistant">
                <div class="message-content">
                    👋 Hello! I'm your AI HR Assistant. I can help you with:
                    <br><br>
                    <span class="info-badge">👤 Employee Information</span>
                    <span class="info-badge">🏖️ Leave Balances</span>
                    <span class="info-badge">💼 Interview Questions</span>
                    <span class="info-badge">📋 Company Policies</span>
                    <br><br>
                    <strong>Sample Employee IDs:</strong> E12345, E12346, E12347
                    <br><br>
                    How can I assist you today?
                </div>
            </div>
            <div class="typing-indicator" id="typingIndicator">
                <span></span><span></span><span></span>
            </div>
        </div>
        <div class="input-container">
            <input type="text" id="userInput" placeholder="Type your message here...">
            <button class="btn btn-primary" id="sendBtn">Send</button>
            <button class="btn btn-secondary" id="clearBtn">Clear</button>
        </div>
    </div>

    <script>
        const chatContainer = document.getElementById('chatContainer');
        const userInput = document.getElementById('userInput');
        const sendBtn = document.getElementById('sendBtn');
        const clearBtn = document.getElementById('clearBtn');
        const typingIndicator = document.getElementById('typingIndicator');

        // Quick action buttons
        document.getElementById('q1').addEventListener('click', () => quickQuestion('Check my leave balance for E12345'));
        document.getElementById('q2').addEventListener('click', () => quickQuestion('Get employee details for E12346'));
        document.getElementById('q3').addEventListener('click', () => quickQuestion('Generate interview questions for Data Scientist'));
        document.getElementById('q4').addEventListener('click', () => quickQuestion('What is the remote work policy?'));

        function quickQuestion(question) {
            userInput.value = question;
            sendMessage();
        }

        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        sendBtn.addEventListener('click', sendMessage);
        clearBtn.addEventListener('click', clearChat);

        function addMessage(content, isUser) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.innerHTML = content.replace(/\\n/g, '<br>');
            messageDiv.appendChild(contentDiv);
            chatContainer.insertBefore(messageDiv, typingIndicator);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        async function sendMessage() {
            const message = userInput.value.trim();
            if (!message) return;

            addMessage(message, true);
            userInput.value = '';
            sendBtn.disabled = true;
            typingIndicator.style.display = 'block';
            chatContainer.scrollTop = chatContainer.scrollHeight;

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message })
                });
                const data = await response.json();
                typingIndicator.style.display = 'none';
                addMessage(data.response, false);
            } catch (error) {
                typingIndicator.style.display = 'none';
                addMessage('Sorry, an error occurred: ' + error.message, false);
            } finally {
                sendBtn.disabled = false;
                userInput.focus();
            }
        }

        async function clearChat() {
            if (confirm('Clear conversation?')) {
                await fetch('/clear', { method: 'POST' });
                const welcome = chatContainer.querySelector('.message.assistant');
                chatContainer.innerHTML = '';
                chatContainer.appendChild(welcome);
                chatContainer.innerHTML += '<div class="typing-indicator" id="typingIndicator"><span></span><span></span><span></span></div>';
            }
        }

        userInput.focus();
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        session_id = request.remote_addr
        
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        
        messages = chat_sessions[session_id] + [HumanMessage(content=user_message)]
        response = agent_executor.invoke({"messages": messages})
        assistant_messages = [msg for msg in response["messages"] if isinstance(msg, AIMessage)]
        assistant_message = assistant_messages[-1].content if assistant_messages else "I apologize, but I couldn't process your request."
        
        chat_sessions[session_id].append(HumanMessage(content=user_message))
        chat_sessions[session_id].append(AIMessage(content=assistant_message))
        
        if len(chat_sessions[session_id]) > 10:
            chat_sessions[session_id] = chat_sessions[session_id][-10:]
        
        return jsonify({'response': assistant_message})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'response': f'Error: {str(e)}'}), 500

@app.route('/clear', methods=['POST'])
def clear():
    session_id = request.remote_addr
    if session_id in chat_sessions:
        chat_sessions[session_id] = []
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    print("🚀 Starting AI-Powered HR Assistant...")
    if OPENAI_API_KEY == "<YOUR_OPENAI_API_KEY>":
        print("📝 Warning: OpenAI API key not set. Set OPENAI_API_KEY environment variable.")
    else:
        print("🔒 OpenAI API key loaded from environment.")
    print("🌐 Open: http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)