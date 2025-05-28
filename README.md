# 🔬🧮 Physics & Math AI Tutor

An intelligent chatbot built with LangGraph and Google Gemini that specializes in answering physics and mathematics questions with detailed explanations and step-by-step solutions.

## 🌟 Features

- **Smart Classification**: Automatically classifies questions into physics or math categories
- **Expert Agents**: Specialized agents for physics and mathematics with comprehensive knowledge
- **Interactive UI**: Clean Streamlit web interface for seamless conversations
- **Step-by-step Solutions**: Detailed explanations with proper mathematical/physical reasoning
- **Multi-domain Coverage**: Supports wide range of topics in both physics and mathematics

## 🏗️ Architecture

The project uses a **3-agent architecture** built with LangGraph:

1. **Classifier Agent**: Determines whether the question is physics or math-related
2. **Router**: Routes the question to the appropriate specialized agent
3. **Subject Agents**: 
   - **Physics Agent**: Handles all physics-related questions
   - **Math Agent**: Handles all mathematics-related questions

## 📊 State Graph Flow

```
START → Classify Message → Router → [Physics Agent | Math Agent] → END
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google API Key (from [Google AI Studio](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/physics-math-ai-tutor.git
cd physics-math-ai-tutor
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_google_api_key_here
```

4. **Run the application:**
```bash
streamlit run physicsmathagent.py
```

5. **Open your browser** and navigate to `http://localhost:8501`

## 📦 Dependencies

```txt
streamlit>=1.28.0
langgraph>=0.0.40
langchain-google-genai>=1.0.0
python-dotenv>=1.0.0
pydantic>=2.0.0
typing-extensions>=4.0.0
```

## 🔧 Usage

### Web Interface
1. Start the Streamlit app
2. Type your physics or math question in the chat input
3. Get detailed, step-by-step explanations from specialized AI agents

### Terminal Mode
Uncomment the terminal mode in the main function for command-line usage:
```python
if __name__ == "__main__":
    run_chatbot_terminal()  # Uncomment this line
    # main()  # Comment this line
```

## 📚 Supported Topics

### Physics Topics
- **Classical Mechanics**: Kinematics, dynamics, energy, momentum
- **Thermodynamics**: Heat transfer, entropy, gas laws
- **Electromagnetism**: Electric fields, magnetic fields, circuits
- **Waves & Optics**: Wave properties, interference, refraction
- **Modern Physics**: Quantum mechanics, relativity
- **Astrophysics**: Planetary motion, cosmology

### Mathematics Topics
- **Algebra**: Linear, polynomial, exponential equations
- **Calculus**: Differential, integral, multivariable calculus
- **Geometry**: Euclidean, analytic geometry, trigonometry
- **Statistics**: Probability, distributions, hypothesis testing
- **Linear Algebra**: Matrices, vector spaces, eigenvalues
- **Discrete Math**: Number theory, combinatorics, graph theory

## 💡 Example Questions

**Physics:**
- "Explain Newton's second law with an example"
- "Calculate the kinetic energy of a 5kg object moving at 10 m/s"
- "What is the photoelectric effect?"

**Mathematics:**
- "Solve the quadratic equation x² + 5x + 6 = 0"
- "Find the derivative of sin(x²)"
- "Explain the concept of limits in calculus"

## 🔄 How It Works

1. **Message Classification**: The classifier agent analyzes the input question and determines if it's physics or math-related using structured output from Google Gemini.

2. **Intelligent Routing**: Based on the classification, the router directs the question to the appropriate specialized agent.

3. **Expert Response**: The specialized agent (Physics or Math) processes the question with domain-specific prompts and provides detailed explanations.

4. **Interactive Display**: The Streamlit interface displays the conversation with proper formatting and maintains chat history.

## 🛠️ Configuration

### Model Selection
The project uses `gemini-1.5-flash` for optimal performance on the free tier. You can modify the model in the code:

```python
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
```

### Custom Prompts
Agent prompts can be customized in the respective agent functions (`physics_agent` and `math_agent`) to adjust the response style and expertise level.

## 📈 Future Enhancements

- [ ] Add support for image/diagram uploads
- [ ] Implement LaTeX rendering for mathematical expressions
- [ ] Add voice input/output capabilities
- [ ] Create specialized sub-agents for advanced topics
- [ ] Add memory for multi-turn conversations
- [ ] Implement user feedback and rating system

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangGraph** for the powerful graph-based conversation framework
- **Google Gemini** for the advanced language model capabilities
- **Streamlit** for the intuitive web interface framework

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/yourusername/physics-math-ai-tutor/issues) page
2. Create a new issue with detailed description
3. Contact the maintainers

---

**Made with ❤️ for students and educators worldwide**
