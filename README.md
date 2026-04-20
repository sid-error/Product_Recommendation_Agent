# Sales Agent Assistant

An intelligent, conversational Sales Assistant built natively using the `google.adk` Python SDK and `Streamlit`. The application securely interfaces with local `Pandas` catalog datasets mapped over volatile memory, ensuring ultra-low latency parsing routines.

## System Architecture

1. **Standalone Conversational LLM**: The orchestrator is entirely scaled back to a singular `Gemini 2.5 Pro` Agent deployed inside Streamlit avoiding arbitrary subprocess pipelines and BrokenResource limits natively associated with FastMCP.
2. **Dynamic Taxonomy Extraction**: A fuzzy-search text extrapolation engine that tokenizes broad user sentences ('liquid Dove soap') directly against intersection matrices inside `product_info.csv` without demanding strict categorical taxonomy responses. 
3. **HTML Rendering**: Secure rendering bypass protocols mapping remote Amazon product `<img>` tags over CSS layouts inside Streamlit `st.markdown`, masking URL injection rejections!

## Setup & Installation

### 1. Install Dependencies
Navigate into the application root and activate a virtual environment, then install requirements:
```bash
cd Sales_Agent
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Rename `.env.example` to `.env` and fill it with your real credentials:
```env
GEMINI_API_KEY=your_actual_api_key_here
```

## Running the Application
To run the Streamlit user interface (which boots the Google ADK Runner framework natively in-memory):
```bash
streamlit run src\ui\app.py
```
