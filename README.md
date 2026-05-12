# GroundWork

## Offline-first clinical AI workflows for Community Health Workers

GroundWork is an interoperable healthcare workflow system designed for low-resource and low-connectivity environments.

Instead of another healthcare chatbot, GroundWork provides:

- multilingual clinical triage
- medical image analysis
- prescription OCR
- FHIR-native interoperability
- offline-first synchronization
- MCP-powered orchestration

---

# The Problem

Community Health Workers (CHWs) often operate with:

- poor internet connectivity
- handwritten prescriptions
- multilingual symptom descriptions
- fragmented healthcare systems
- limited infrastructure
- delayed synchronization

Most healthcare AI systems assume:

- stable internet
- centralized hospital infrastructure
- English-only workflows
- structured EHR environments

These assumptions fail in real frontline healthcare delivery.

---

# Our Solution

GroundWork acts as a lightweight healthcare orchestration layer for frontline healthcare delivery.

A CHW can:

- upload a prescription image
- send symptoms in their local language
- analyze medical device images
- continue working offline
- sync data later

The frontend only exposes:

- urgency
- findings
- next steps
- sync status

while the backend handles:

- MCP workflows
- OCR pipelines
- terminology normalization
- FHIR generation
- offline queueing
- synchronization retries

---

# Core Features

## Multilingual Clinical Triage

GroundWork extracts structured clinical data from multilingual free-text input.

Supported languages include:

- Hindi
- Tamil
- Telugu
- Malayalam
- Kannada
- Bengali
- Arabic
- Swahili
- Tagalog
- English
- mixed-language clinical text

The triage pipeline extracts:

- symptoms
- vitals
- duration
- severity score
- referral urgency
- language metadata

The system combines LLM extraction with deterministic severity rules for safer workflows.

### Example

Input:

```text
"Mera sir dukh raha hai aur BP 170/110 hai"
```

Extracted:

```json
{
  "symptoms": ["headache"],
  "severity_score": 0.88,
  "referral_flag": true
}
```

---

## Medical Image Analysis

GroundWork supports direct analysis of clinical images captured by CHWs.

Supported image workflows:

- thermometer readings
- pulse oximeters
- glucometers
- malaria RDT strips
- HIV RDT strips
- wounds
- edema
- rashes
- referral records

The system extracts:

- vitals
- RDT results
- visible findings
- OCR text
- device measurements

without unsupported diagnostic inference.

### Example

Pulse oximeter image:

```json
{
  "spo2": 88,
  "pulse": 122,
  "severity_score": 0.91,
  "referral_flag": true
}
```

---

## Prescription OCR

GroundWork implements a two-stage OCR pipeline:

```text
Prescription Image
        ↓
OCR Transcription
        ↓
Structured Clinical Extraction
        ↓
Normalization
        ↓
FHIR Preparation
```

Supported documents:

- handwritten prescriptions
- referral slips
- lab reports
- health records

The OCR pipeline separates:

1. raw transcription
2. structured extraction

to reduce hallucinations and preserve auditability.

---

## FHIR Interoperability

GroundWork generates FHIR R4 Transaction Bundles automatically.

Generated resources include:

- Observation
- Condition
- MedicationRequest
- RiskAssessment

LOINC codes are attached to vitals automatically.

SNOMED CT concepts are generated for symptoms and findings.

RxNorm resolution is used for medications.

---

## Offline-First Infrastructure

GroundWork is designed around unreliable connectivity.

Features include:

- SQLite queueing
- deferred synchronization
- retry logic
- exponential backoff
- persistent offline storage
- idempotent dispatch

Possible sync states:

- Saved offline
- Sync pending
- Synced
- Failed

If synchronization fails, bundles are stored locally and retried automatically by a background worker.

---

# Architecture

```text
                ┌──────────────────────┐
                │ CHW / Mobile Client  │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Prompt Opinion Agent │
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

# MCP Tooling

GroundWork exposes healthcare workflows through FastMCP tools.

Core tools include:

| Tool | Purpose |
|---|---|
| `extract_triage` | multilingual text triage |
| `extract_triage_with_history` | triage with FHIR history context |
| `analyze_clinical_image` | clinical image analysis |
| `triage_document_image` | OCR + document extraction |
| `build_fhir_bundle` | generate FHIR bundles |
| `dispatch_fhir_bundle` | dispatch bundles with retry logic |
| `check_sync_queue` | queue monitoring |

---

# Workflow Pipelines

## Text Triage Pipeline

```text
Clinical Text
      ↓
LLM Extraction
      ↓
Normalization
      ↓
Deterministic Severity Rules
      ↓
FHIR Generation
```

---

## Vision Pipeline

```text
Clinical Image
      ↓
Vision Model Extraction
      ↓
Vital Parsing
      ↓
LOINC Mapping
      ↓
FHIR Preparation
```

---

## OCR Pipeline

```text
Prescription Image
        ↓
OCR Extraction
        ↓
Medication Parsing
        ↓
RxNorm Resolution
        ↓
MedicationRequest Bundle
```

---

# Offline Queueing + Synchronization

GroundWork persists failed dispatches locally using SQLite.

The queue stores:

- FHIR bundles
- retry attempts
- timestamps
- CHW IDs
- synchronization state
- idempotency keys

Synchronization architecture:

```text
Dispatch Failure
       ↓
SQLite Queue
       ↓
Background Retry Worker
       ↓
FHIR Synchronization
```

The dispatcher includes:

- exponential backoff
- retry policies
- duplicate prevention
- deferred synchronization

---

# Terminology Layer

GroundWork dynamically resolves healthcare terminology using:

- SNOMED CT
- LOINC
- RxNorm

The terminology subsystem includes:

- SQLite caching
- offline seed vocabularies
- API fallback resolution
- in-memory hot cache

This allows partial offline interoperability even in disconnected environments.

---

# Tech Stack

## Frontend

- React

## Backend

- FastAPI
- FastMCP
- SQLite

## AI Pipelines

- Groq API
- Llama 3.3 70B
- Llama 4 Vision

## Healthcare Standards

- MCP
- A2A
- FHIR R4
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
│   ├── sync_queue.py
│   └── fhir_context.py
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

# Example Workflow

```text
CHW uploads pulse oximeter image
            ↓
GroundWork extracts:
    SpO2 = 88%
    Pulse = 122
            ↓
Severity rules trigger referral
            ↓
FHIR Bundle generated
            ↓
Stored offline if no connectivity
            ↓
Later synchronized automatically
```

---

# What Makes GroundWork Different

GroundWork is not:

- a generic chatbot
- a hospital dashboard
- an EHR clone

GroundWork focuses on:

- deployability
- interoperability
- offline resilience
- operational simplicity
- low cognitive overhead
- frontline healthcare workflows

The goal is to make complex healthcare infrastructure feel effortless for Community Health Workers.

---

# Future Plans

- expanded multilingual support
- longitudinal patient history
- WhatsApp integration for LMIC workflows
- edge-device deployment
- deeper FHIR integrations
- public health deployment pilots
- lightweight edge inference

---

# Closing Statement

GroundWork transforms fragmented frontline healthcare workflows into interoperable, offline-resilient clinical infrastructure.

By combining:

- multilingual AI extraction
- clinical image understanding
- prescription OCR
- FHIR-native interoperability
- offline synchronization
- MCP orchestration

GroundWork enables Community Health Workers to operate effectively in environments where traditional healthcare systems fail.