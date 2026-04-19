from fastmcp import FastMCP
from dotenv import load_dotenv
import asyncio

# Load environment variables FIRST so the modules can access the API key
load_dotenv()

# Now import your custom logic
from modules.triage import triage_extractor

mcp = FastMCP(
    name="groundwork",
    instructions="GroundWork: Community health triage and FHIR bundle generation."
)

@mcp.tool()
async def extract_triage(raw_text: str, patient_id: str) -> dict:
    """Extract triage data from Hinglish clinical text."""
    # This calls your new Gemini 3.1 Flash Lite logic
    return await triage_extractor(raw_text, patient_id)

@mcp.tool()
def build_fhir_bundle(triage_output: dict, patient_id: str) -> dict:
    """Build a FHIR Transaction Bundle from triage output."""
    # We will wire this up in Phase 4
    return {
        "status": "stub",
        "message": "FHIR logic pending Phase 4",
        "patient_id": patient_id
    }

if __name__ == "__main__":
    # Using 0.0.0.0 is fine, but 127.0.0.1 is usually more stable for ngrok on Windows
    mcp.run(transport="sse", host="127.0.0.1", port=8000)