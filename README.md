# AI-AEC Interpreter Layer: MVP

> **Thesis Prototype:** An agentic interface that bridges unstructured site data with structured IFC/BIM models.

This project serves as a Proof of Concept (MVP) for a Master Thesis in Computational Design. It establishes an **"Interpreter Layer"** that leverages **Graph-RAG** and **LLM Tool Calling** to accurately map unstructured site reports (natural language) to specific IFC elements (GUIDs).

## Project Structure

```
mscd_demo/
├── data/
│   ├── ifc/                    # IFC model files
│   ├── images/                 # Site photos and renders
│   ├── texts/                  # Issue reports and chat logs
│   └── schemas/                # PPVC / CORENET-X templates (JSON/CSV)
├── src/
│   ├── ifc_engine.py           # IFC graph indexing and querying
│   ├── agent_tools.py          # LangChain tool definitions
│   └── main.py                 # Agent orchestration
├── .env                        # API key configuration
└── requirements.txt            # Python dependencies
```

## System Logic

Unlike traditional keyword search, this system is built on **IFC Topological Semantics**.

### Architecture

**1. Semantic Graph Indexing** (`ifc_engine.py`)
- Parses the `.ifc` file using `IfcOpenShell`
- Reconstructs the spatial topology tree: `Site` → `Storey` → `Space` → `Elements`
- Avoids feeding raw IFC text to the LLM; instead provides a precise semantic query interface

**2. Agent Orchestration** (`main.py`)
- Powered by **LangChain** and **"gemini-2.5-flash"**
- Uses a **ReAct / Tool-Calling** pattern: Intent analysis → Tool selection → Data retrieval → GUID isolation

**3. Tool Bridge** (`agent_tools.py`)
- `get_elements_by_room`: Executes topological queries (e.g., "What is in the Kitchen?")
- `get_element_details`: Extracts Property Sets (Psets) for compliance verification

## Usage

### Prerequisites

- Python 3.9+
- Google Gemini API Key

### Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
# Create a .env file in the root directory with your API key:
# GOOGLE_API_KEY=your_api_key_here
```

### Running the Agent

Execute the main script to simulate receiving unstructured text from a construction site:

```bash
python src/main.py
```

### Example Workflow

**Input:** "Report: Significant water pooling observed on the floor slab in the kitchen area."

**Agent Action:**
1. Identifies context: "Kitchen"
2. Calls tool: `get_elements_by_room("kitchen")`
3. Retrieves elements: `[Wall_01, Slab_05, Sink_02...]`
4. Matches "floor slab" description to `Slab_05` (Type: `IfcSlab`)

**Output:** Returns the specific **GUID** and a structured issue report

## Future Roadmap

- **Visual Alignment**: Integrate CLIP models to enable image-to-IFC retrieval using site photos
- **Compliance**: Incorporate CORENET-X rule checks for automated compliance validation
- **Feedback Loop**: Write identified issues back into the IFC file using the BCF (BIM Collaboration Format)
