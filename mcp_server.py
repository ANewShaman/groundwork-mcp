from fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP(
    name="groundwork",
    instructions="GroundWork: Community health triage and FHIR bundle generation."
)

@mcp.tool()
def extract_triage(raw_text: str, patient_id: str) -> dict:
    """Extract triage data from Hinglish clinical text."""
    return {
        "status": "stub",
        "message": "Connection successful",
        "raw_text": raw_text,
        "patient_id": patient_id
    }

@mcp.tool()
def build_fhir_bundle(triage_output: dict, patient_id: str) -> dict:
    """Build a FHIR Transaction Bundle from triage output."""
    return {
        "status": "stub",
        "message": "FHIR logic pending Phase 4",
        "patient_id": patient_id
    }

if __name__ == "__main__":
    # In FastMCP 3.x, this is the proper way to run the SSE server
    # We use 127.0.0.1 to avoid the 'actively refused' Windows/IPv6 issue
    mcp.run(transport="sse", host="127.0.0.1", port=8000)