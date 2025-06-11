# muilti_Agent_Sys
git init
git add .
git commit -m "Initial commit: Multi-agent Tutor Agent"
git branch -M main
git remote add origin https://github.com/padminK/ai-tutor-multiagent.git
git push -u origin main
# AI Tutor Multi-Agent System

A multi-agent AI tutoring assistant that intelligently delegates questions to specialized sub-agents (Math, Physics, Chemistry, General). Built using Python and inspired by Google's Agent Development Kit (ADK) concepts.

---

## Features

- **Multi-Agent Delegation:** Tutor agent routes queries to the best specialist agent.
- **Specialist Agents:** Math, Physics, Chemistry, and General agents.
- **Tool Usage:** Each agent can use tools (calculator, formula lookup, Gemini API).
- **Gemini API Integration:** GeneralAgent uses Gemini for open-ended questions.
- **CLI Demo & Interactive Modes:** Try out the system in your terminal.

---

## Setup

1. **Clone the repository:**
   
   git clone https://github.com/padminK/ai-tutor-multiagent.git
   cd ai-tutor-multiagent
   Install Dependencies
   pip install -r requirements.txt
   import os
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

set GEMINI_API_KEY=AIzaSyBEt0HbtkTEHIAYCj_uGGvMiytrzD_7Is4
##usage 
python tutor_Agent.py
You will be prompted to choose a mode:

Interactive Mode: Type your own questions.
Demo Mode: See sample questions and answers.
Example questions:

Calculate 2^3 + sqrt(16)
What is the formula for force?
Tell me about carbon
What is the atomic number of oxygen?
##Project Structure
ai-tutor-multiagent/
│
├── tutor_Agent.py
├── requirements.txt
├── README.md
└── .gitignore
##Deployment
To deploy on Railway or Vercel:

Wrap the logic in a FastAPI or Flask server.
Expose endpoints for question answering.
Store your Gemini API key securely (use environment variables).
##License
MIT
##Author
padminK
