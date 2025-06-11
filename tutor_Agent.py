"""
Multi-Agent Tutoring System
Based on Google's Agent Development Kit (ADK) concepts
"""

import json
import re
import math
import asyncio
from typing import Dict, List, Any, Optional, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentType(Enum):
    TUTOR = "tutor"
    MATH = "math"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    GENERAL = "general"

@dataclass
class Message:
    content: str
    sender: str
    recipient: str
    metadata: Dict[str, Any] = None

@dataclass
class TaskResult:
    success: bool
    result: str
    confidence: float
    tools_used: List[str] = None

class Tool(ABC):
    """Abstract base class for agent tools"""
    
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def description(self) -> str:
        pass
    
    @abstractmethod
    async def execute(self, input_data: str) -> str:
        pass

class CalculatorTool(Tool):
    """Calculator tool for mathematical operations"""
    
    def name(self) -> str:
        return "calculator"
    
    def description(self) -> str:
        return "Performs basic mathematical calculations including arithmetic, trigonometry, and algebra"
    
    async def execute(self, expression: str) -> str:
        try:
            # Clean the expression
            expression = expression.strip()
            
            # Handle common mathematical functions
            expression = expression.replace("sin", "math.sin")
            expression = expression.replace("cos", "math.cos")
            expression = expression.replace("tan", "math.tan")
            expression = expression.replace("log", "math.log")
            expression = expression.replace("sqrt", "math.sqrt")
            expression = expression.replace("pi", "math.pi")
            expression = expression.replace("e", "math.e")
            
            # Evaluate the expression safely
            allowed_names = {
                k: v for k, v in math.__dict__.items() if not k.startswith("__")
            }
            allowed_names.update({"__builtins__": {}})
            
            result = eval(expression, allowed_names)
            return f"Result: {result}"
        except Exception as e:
            return f"Error in calculation: {str(e)}"

class PhysicsFormulaTool(Tool):
    """Physics formula calculator"""
    
    def name(self) -> str:
        return "physics_calculator"
    
    def description(self) -> str:
        return "Calculates physics problems using common formulas"
    
    async def execute(self, problem: str) -> str:
        try:
            # Simple physics formulas
            formulas = {
                "velocity": "v = d/t",
                "acceleration": "a = (v_f - v_i)/t",
                "force": "F = m*a",
                "energy": "E = m*c^2",
                "power": "P = W/t",
                "pressure": "P = F/A"
            }
            
            problem_lower = problem.lower()
            for formula_type, formula in formulas.items():
                if formula_type in problem_lower:
                    return f"Formula for {formula_type}: {formula}"
            
            return "Please specify the type of physics problem (velocity, force, energy, etc.)"
        except Exception as e:
            return f"Error in physics calculation: {str(e)}"

class ChemistryTool(Tool):
    """Chemistry reference tool"""
    
    def name(self) -> str:
        return "chemistry_reference"
    
    def description(self) -> str:
        return "Provides chemical formulas, molecular weights, and basic chemistry information"
    
    async def execute(self, query: str) -> str:
        try:
            # Basic chemistry data
            elements = {
                "hydrogen": {"symbol": "H", "atomic_number": 1, "atomic_weight": 1.008},
                "helium": {"symbol": "He", "atomic_number": 2, "atomic_weight": 4.003},
                "carbon": {"symbol": "C", "atomic_number": 6, "atomic_weight": 12.011},
                "nitrogen": {"symbol": "N", "atomic_number": 7, "atomic_weight": 14.007},
                "oxygen": {"symbol": "O", "atomic_number": 8, "atomic_weight": 15.999},
                "sodium": {"symbol": "Na", "atomic_number": 11, "atomic_weight": 22.990},
                "chlorine": {"symbol": "Cl", "atomic_number": 17, "atomic_weight": 35.453}
            }
            
            query_lower = query.lower()
            for element, data in elements.items():
                if element in query_lower:
                    return f"{element.title()}: Symbol={data['symbol']}, Atomic Number={data['atomic_number']}, Atomic Weight={data['atomic_weight']}"
            
            return "Please specify an element name for information"
        except Exception as e:
            return f"Error in chemistry lookup: {str(e)}"

class Agent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, agent_id: str, agent_type: AgentType):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.tools: List[Tool] = []
        self.capabilities: List[str] = []
        
    def add_tool(self, tool: Tool):
        self.tools.append(tool)
        
    @abstractmethod
    async def process_request(self, message: Message) -> TaskResult:
        pass
    
    def can_handle(self, request: str) -> float:
        """Returns confidence score (0-1) for handling the request"""
        return 0.0

class MathAgent(Agent):
    """Specialized agent for mathematics"""
    
    def __init__(self):
        super().__init__("math_agent", AgentType.MATH)
        self.add_tool(CalculatorTool())
        self.capabilities = [
            "arithmetic", "algebra", "calculus", "geometry", 
            "trigonometry", "statistics", "probability"
        ]
    
    def can_handle(self, request: str) -> float:
        math_keywords = [
            "calculate", "solve", "equation", "formula", "math", "mathematics",
            "add", "subtract", "multiply", "divide", "integral", "derivative",
            "sin", "cos", "tan", "log", "sqrt", "algebra", "geometry"
        ]
        
        request_lower = request.lower()
        matches = sum(1 for keyword in math_keywords if keyword in request_lower)
        return min(matches * 0.2, 1.0)
    
    async def process_request(self, message: Message) -> TaskResult:
        try:
            request = message.content
            
            # Check if calculation is needed
            if any(op in request for op in ['+', '-', '*', '/', '=', 'calculate', 'solve']):
                # Extract mathematical expression
                calc_tool = self.tools[0]  # Calculator tool
                result = await calc_tool.execute(request)
                
                return TaskResult(
                    success=True,
                    result=f"Math Agent: {result}",
                    confidence=0.9,
                    tools_used=[calc_tool.name()]
                )
            else:
                # Provide mathematical guidance
                return TaskResult(
                    success=True,
                    result=f"Math Agent: I can help with mathematical calculations and problem-solving. Please provide a specific equation or calculation.",
                    confidence=0.7,
                    tools_used=[]
                )
                
        except Exception as e:
            return TaskResult(
                success=False,
                result=f"Math Agent Error: {str(e)}",
                confidence=0.0,
                tools_used=[]
            )

class PhysicsAgent(Agent):
    """Specialized agent for physics"""
    
    def __init__(self):
        super().__init__("physics_agent", AgentType.PHYSICS)
        self.add_tool(PhysicsFormulaTool())
        self.add_tool(CalculatorTool())
        self.capabilities = [
            "mechanics", "thermodynamics", "electromagnetism", 
            "optics", "quantum", "relativity"
        ]
    
    def can_handle(self, request: str) -> float:
        physics_keywords = [
            "physics", "force", "velocity", "acceleration", "energy", "power",
            "momentum", "wave", "frequency", "electric", "magnetic", "gravity",
            "pressure", "temperature", "heat", "light", "quantum"
        ]
        
        request_lower = request.lower()
        matches = sum(1 for keyword in physics_keywords if keyword in request_lower)
        return min(matches * 0.25, 1.0)
    
    async def process_request(self, message: Message) -> TaskResult:
        try:
            request = message.content
            
            # Use physics formula tool
            physics_tool = self.tools[0]  # Physics formula tool
            result = await physics_tool.execute(request)
            
            return TaskResult(
                success=True,
                result=f"Physics Agent: {result}",
                confidence=0.85,
                tools_used=[physics_tool.name()]
            )
                
        except Exception as e:
            return TaskResult(
                success=False,
                result=f"Physics Agent Error: {str(e)}",
                confidence=0.0,
                tools_used=[]
            )

class ChemistryAgent(Agent):
    """Specialized agent for chemistry"""
    
    def __init__(self):
        super().__init__("chemistry_agent", AgentType.CHEMISTRY)
        self.add_tool(ChemistryTool())
        self.capabilities = [
            "organic", "inorganic", "analytical", "physical", 
            "biochemistry", "periodic_table"
        ]
    
    def can_handle(self, request: str) -> float:
        chemistry_keywords = [
            "chemistry", "chemical", "element", "compound", "reaction", "molecule",
            "atom", "periodic", "bond", "organic", "inorganic", "ph", "acid", "base"
        ]
        
        request_lower = request.lower()
        matches = sum(1 for keyword in chemistry_keywords if keyword in request_lower)
        return min(matches * 0.25, 1.0)
    
    async def process_request(self, message: Message) -> TaskResult:
        try:
            request = message.content
            
            # Use chemistry tool
            chem_tool = self.tools[0]  # Chemistry reference tool
            result = await chem_tool.execute(request)
            
            return TaskResult(
                success=True,
                result=f"Chemistry Agent: {result}",
                confidence=0.8,
                tools_used=[chem_tool.name()]
            )
                
        except Exception as e:
            return TaskResult(
                success=False,
                result=f"Chemistry Agent Error: {str(e)}",
                confidence=0.0,
                tools_used=[]
            )

class GeneralAgent(Agent):
    """General purpose agent for non-specialized queries"""
    
    def __init__(self):
        super().__init__("general_agent", AgentType.GENERAL)
        self.capabilities = ["general_knowledge", "conversation", "guidance"]
    
    def can_handle(self, request: str) -> float:
        return 0.3  # Always provides a fallback option
    
    async def process_request(self, message: Message) -> TaskResult:
        try:
            # Use Gemini for general questions
            genai.configure(api_key="YOUR_GEMINI_API_KEY")
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(message.content)
            return TaskResult(
                success=True,
                result=f"General Agent: {response.text}",
                confidence=0.5,
                tools_used=["Gemini"]
            )
        except Exception as e:
            return TaskResult(
                success=False,
                result=f"General Agent Error: {str(e)}",
                confidence=0.0,
                tools_used=[]
            )

class TutorAgent:
    """Main tutor agent that orchestrates specialized agents"""
    
    def __init__(self):
        self.agent_id = "tutor_agent"
        self.agents: List[Agent] = []
        self.conversation_history: List[Message] = []
        
        # Initialize specialized agents
        self.agents.append(MathAgent())
        self.agents.append(PhysicsAgent())
        self.agents.append(ChemistryAgent())
        self.agents.append(GeneralAgent())
        
        logger.info(f"Tutor Agent initialized with {len(self.agents)} specialized agents")
    
    def find_best_agent(self, request: str) -> Agent:
        """Find the most suitable agent for the request"""
        agent_scores = []
        
        for agent in self.agents:
            confidence = agent.can_handle(request)
            agent_scores.append((agent, confidence))
            logger.info(f"Agent {agent.agent_id} confidence: {confidence}")
        
        # Sort by confidence and return the best agent
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        best_agent = agent_scores[0][0]
        
        logger.info(f"Selected agent: {best_agent.agent_id}")
        return best_agent
    
    async def process_query(self, user_query: str) -> str:
        """Main method to process user queries"""
        try:
            logger.info(f"Processing query: {user_query}")
            
            # Create message
            message = Message(
                content=user_query,
                sender="user",
                recipient="tutor"
            )
            
            # Add to conversation history
            self.conversation_history.append(message)
            
            # Find best agent
            best_agent = self.find_best_agent(user_query)
            
            # Process request with selected agent
            result = await best_agent.process_request(message)
            
            # Create response message
            response_message = Message(
                content=result.result,
                sender=best_agent.agent_id,
                recipient="user"
            )
            
            self.conversation_history.append(response_message)
            
            # Format response
            response = f"{result.result}\n"
            if result.tools_used:
                response += f"[Tools used: {', '.join(result.tools_used)}]"
            
            logger.info(f"Response generated with confidence: {result.confidence}")
            return response
            
        except Exception as e:
            error_msg = f"Tutor Agent Error: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def get_agent_capabilities(self) -> Dict[str, List[str]]:
        """Return capabilities of all agents"""
        capabilities = {}
        for agent in self.agents:
            capabilities[agent.agent_id] = agent.capabilities
        return capabilities
    
    def get_conversation_history(self) -> List[Dict]:
        """Return formatted conversation history"""
        return [
            {
                "sender": msg.sender,
                "recipient": msg.recipient,
                "content": msg.content
            }
            for msg in self.conversation_history
        ]

class TutorApp:
    """Main application class"""
    
    def __init__(self):
        self.tutor = TutorAgent()
        self.running = True
    
    def display_welcome(self):
        """Display welcome message and capabilities"""
        print("=" * 60)
        print("🤖 AI TUTOR - Multi-Agent Learning Assistant")
        print("=" * 60)
        print("\nAvailable Specialists:")
        
        capabilities = self.tutor.get_agent_capabilities()
        for agent_id, caps in capabilities.items():
            agent_name = agent_id.replace("_", " ").title()
            print(f"📚 {agent_name}: {', '.join(caps)}")
        
        print("\n" + "=" * 60)
        print("Type your questions or 'quit' to exit")
        print("Examples:")
        print("- 'Calculate 2^3 + sqrt(16)'")
        print("- 'What is the formula for force?'")
        print("- 'Tell me about carbon'")
        print("=" * 60 + "\n")
    
    async def run_interactive(self):
        """Run interactive mode"""
        self.display_welcome()
        
        while self.running:
            try:
                user_input = input("🎓 Ask me anything: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye! Happy learning!")
                    break
                
                if not user_input:
                    continue
                
                print("🤔 Thinking...")
                response = await self.tutor.process_query(user_input)
                print(f"✅ {response}\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye! Happy learning!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}\n")
    
    async def run_demo(self):
        """Run demo with predefined questions"""
        print("🚀 Running Demo Mode...\n")
        
        demo_questions = [
            "Calculate 15 * 8 + sqrt(25)",
            "What is the formula for velocity?",
            "Tell me about oxygen",
            "Solve for x: 2x + 5 = 15",
            "What is force in physics?",
            "What is the atomic number of carbon?"
        ]
        
        for question in demo_questions:
            print(f"❓ Question: {question}")
            response = await self.tutor.process_query(question)
            print(f"✅ Answer: {response}\n")
            await asyncio.sleep(1)  # Small delay for readability

async def main():
    """Main function"""
    app = TutorApp()
    
    print("Choose mode:")
    print("1. Interactive Mode")
    print("2. Demo Mode")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            await app.run_interactive()
        elif choice == "2":
            await app.run_demo()
        else:
            print("Invalid choice. Running interactive mode...")
            await app.run_interactive()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    asyncio.run(main())
