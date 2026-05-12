# GroundWork

## Offline-first clinical AI workflows for Community Health Workers

GroundWork is an interoperable healthcare workflow system designed for low-resource and low-connectivity environments.

Instead of another healthcare chatbot, GroundWork focuses on real frontline healthcare workflows:

- multilingual clinical triage
- medical image analysis
- prescription OCR
- FHIR-native interoperability
- offline-first synchronization
- MCP-powered orchestration

---

# Why GroundWork?

Community Health Workers (CHWs) often work with:

- poor connectivity
- handwritten prescriptions
- multilingual patient communication
- fragmented healthcare systems
- delayed synchronization
- limited infrastructure

Most healthcare AI systems assume:
- stable internet
- centralized hospital infrastructure
- structured EHR environments
- English-only workflows

GroundWork is designed for environments where those assumptions fail.

---

# Core Features

## Multilingual Clinical Triage

GroundWork extracts structured clinical data from multilingual text and voice workflows.

Supported outputs include:
- symptoms
- vitals
- severity scoring
- referral urgency
- workflow flags
- language metadata

Example:

```json
{
  "symptoms": ["headache"],
  "severity_score": 0.88,
  "referral_flag": true
}
```

---

## Medical Image Analysis

GroundWork supports direct analysis of frontline clinical images:

- pulse oximeters
- glucometers
- thermometers
- HIV RDT strips
- malaria RDT strips
- wound images
- edema/rashes
- referral records

The system extracts:
- vitals
- device measurements
- RDT results
- OCR text
- workflow observations

without unsupported diagnostic inference.

---

## Prescription OCR

GroundWork implements a structured OCR pipeline for:

- handwritten prescriptions
- referral slips
- multilingual medication records
- clinical documents

Pipeline:

```text
Prescription Image
        ↓
OCR Extraction
        ↓
Structured Parsing
        ↓
Normalization
        ↓
FHIR Bundle Generation
```

---

## FHIR-Native Interoperability

GroundWork automatically generates FHIR R4 transaction bundles.

Supported resources:
- Observation
- MedicationRequest
- Condition
- RiskAssessment

Terminology support includes:
- LOINC
- SNOMED CT
- RxNorm

---

## Offline-First Synchronization

GroundWork is designed around unreliable connectivity.

Features include:
- SQLite queueing
- deferred synchronization
- retry logic
- idempotent dispatching
- persistent offline storage

If synchronization fails:
1. bundles are stored locally
2. retry workers attempt synchronization later
3. duplicate bundles are prevented automatically

---

# Example Output

Generated directly from pulse oximeter image extraction and FHIR conversion.

```json
{
  "resourceType": "Observation",
  "status": "final",
  "subject": {
    "reference": "Patient/pt-rx-001"
  },
  "code": {
    "coding": [
      {
        "system": "http://loinc.org",
        "code": "59408-5"
      }
    ]
  },
  "valueQuantity": {
    "value": 99,
    "unit": "%",
    "system": "http://unitsofmeasure.org",
    "code": "%"
  }
}
```

---

# End-to-End Workflow

```text
CHW uploads pulse oximeter image
            ↓
GroundWork extracts:
    SpO2 = 99%
    Pulse = 68 bpm
            ↓
FHIR Observation generated
            ↓
Bundle queued locally if offline
            ↓
Automatic synchronization retry
```

---

# Architecture

```text
                ┌──────────────────────┐
                │ CHW / Mobile Client  │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ GroundWork MCP Server│
                └──────────┬───────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼

┌──────────────┐  ┌────────────────┐  ┌────────────────┐
│ Text Triage  │  │ Vision Triage │  │ OCR Pipeline   │
└──────┬───────┘  └──────┬─────────┘  └──────┬─────────┘
       ▼                 ▼                   ▼

┌──────────────────────────────────────────────┐
│ Normalization + Terminology Resolution       │
└────────────────────┬─────────────────────────┘
                     ▼

          ┌────────────────────────┐
          │ FHIR Bundle Generation │
          └────────────┬───────────┘
                       ▼

          ┌────────────────────────┐
          │ Offline Sync Queue     │
          └────────────┬───────────┘
                       ▼

          ┌────────────────────────┐
          │ FHIR Server / EHR      │
          └────────────────────────┘
```

---

# MCP Integration (Prompt Opinion Platform)

GroundWork exposes healthcare workflows through MCP tools using FastMCP.

To connect GroundWork to the Prompt Opinion platform:

## 1. Start the GroundWork MCP server

```bash
python main.py
```

Example local server:

```text
http://localhost:8000
```

---

## 2. Expose the local server using ngrok

```bash
ngrok http 8000
```

Example output:

```text
Forwarding https://abc123.ngrok-free.app -> http://localhost:8000
```

---

## 3. Add MCP Server in Prompt Opinion

Inside the Prompt Opinion platform:

- Open MCP Configuration
- Add new MCP server
- Paste the ngrok HTTPS endpoint

Example:

```text
https://abc123.ngrok-free.app
```

GroundWork MCP tools will now become available inside Prompt Opinion workflows.

---

## Available MCP Tools

| Tool | Purpose |
|---|---|
| `extract_triage` | multilingual text triage |
| `analyze_clinical_image` | clinical image analysis |
| `triage_document_image` | OCR + prescription extraction |
| `build_fhir_bundle` | generate FHIR transaction bundles |
| `dispatch_fhir_bundle` | queue-aware FHIR dispatch |
| `check_sync_queue` | offline queue monitoring |

---

# What GroundWork Demonstrates

- Multimodal clinical extraction
- Multilingual prescription OCR
- Medical device reading interpretation
- Offline-first healthcare workflows
- FHIR-native interoperability
- LOINC / SNOMED / RxNorm normalization
- SQLite-backed synchronization queues
- Idempotent retry semantics
- CHW-oriented workflow design

---

# Tech Stack

## Backend
- FastAPI
- FastMCP
- SQLite

## AI Pipelines
- Groq API
- Llama 3.3 70B
- Llama 4 Vision

## Standards
- FHIR R4
- MCP
- SNOMED CT
- LOINC
- RxNorm

---

# Repository Structure

```text
groundwork/
│
├── modules/
│   ├── triage.py
│   ├── vision_triage.py
│   ├── ocr_bridge.py
│   ├── fhir_ops.py
│   ├── terminology.py
│   ├── normalize.py
│   ├── inference_engine.py
│   ├── action_dispatcher.py
│   └── sync_queue.py
│
├── groundwork_queue.db
├── terminology_cache.db
├── main.py
└── requirements.txt
```

---

# Running Locally

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Environment Variables

```env
GROQ_API_KEY=your_api_key
```

## Start GroundWork

```bash
python main.py
```

---

# Recent End-to-End Demo Results

GroundWork successfully demonstrated:

- 4 generated FHIR Observation resources
- 3 generated MedicationRequest resources
- multilingual Kannada prescription OCR
- pulse oximeter extraction
- glucometer extraction
- thermometer extraction
- offline queue fallback
- SQLite persistence
- retry + idempotency validation

Example merged bundle summary:

| Resource | Count |
|---|---|
| Observation | 4 |
| MedicationRequest | 3 |
| RiskAssessment | 1 |

---

# Vision

GroundWork aims to make interoperable healthcare infrastructure deployable in environments where traditional systems struggle:

- rural healthcare
- low-connectivity clinics
- mobile CHW workflows
- multilingual frontline care
- LMIC healthcare systems

The goal is simple:

Make complex healthcare infrastructure feel lightweight, resilient, and deployable anywhere.