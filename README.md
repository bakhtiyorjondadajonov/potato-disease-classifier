# Plant Disease Classifier API

CNN-based potato disease classification plus **Gemini-powered multi-plant disease analysis** for **Tomato, Corn, Pepper, Apple, and Strawberry**. The CNN model detects **Early Blight**, **Late Blight**, and **Healthy** potato leaves. Gemini 2.5 Flash powers the analysis, advice, crop calendar, and severity endpoints — each plant type uses a dedicated expert prompt with research-backed disease knowledge.

**Base URL:** `http://localhost:8000`

---

## Why This API Exists

Crop diseases cause **10-25% annual yield losses** worldwide. Most farmers and small-scale growers lack access to plant pathologists who can diagnose problems on sight. By the time visible symptoms are bad enough to Google, the disease has often already spread to neighboring plants.

This API solves that problem by letting anyone **photograph a leaf and get an instant diagnosis with actionable treatment advice** — no pathology degree needed. The system combines:

- **Fast CNN classification** for potato (the primary crop) — works offline, sub-second, zero API cost
- **Gemini Vision analysis** for 5 additional crops — broader coverage using expert-prompted AI that knows the specific diseases and visual symptoms for each plant
- **Treatment advice + crop calendars** — so the user knows not just *what's wrong*, but *what to do* and *when to do it*
- **Severity assessment** — so the user knows *how urgently* they need to act

The goal is a complete decision pipeline: **Detect → Diagnose → Advise → Schedule → Prioritize**.

---

## Quick Start

```bash
cd potato-disease-classifier
pip install -r requirements.txt
python main.py
```

The server starts on `http://localhost:8000`. Interactive docs are available at `/docs` (Swagger UI) and `/redoc`.

### Environment Variables

All prefixed with `POTATO_`:

| Variable | Default | Description |
|----------|---------|-------------|
| `POTATO_GEMINI_API_KEY` | `""` | Required for `/advice`, `/crop-calendar`, `/severity`, `/analyze/*` |
| `POTATO_MODEL_PATH` | `./cnn_model.keras` | Path to the CNN model file |
| `POTATO_DEBUG` | `false` | Enable debug logging |

---

## Endpoints Overview

| # | Method | Path | Description | Rate Limit |
|---|--------|------|-------------|------------|
| 1 | GET | `/ping` | Liveness check | None |
| 2 | GET | `/health` | Health status + model loaded | None |
| 3 | POST | `/predict` | Classify potato leaf image (CNN) | 10/min |
| 4 | POST | `/advice` | Get treatment advice for a disease | 5/min |
| 5 | POST | `/crop-calendar` | Get action timeline for a disease | 5/min |
| 6 | POST | `/severity` | Analyze disease severity from image | 5/min |
| 7 | POST | `/analyze/{plant_type}` | Gemini vision disease diagnosis | 5/min |

**Supported plant types for `/analyze`:** `tomato`, `corn`, `pepper`, `apple`, `strawberry`

---

## Business Logic & Feature Rationale

Each endpoint exists for a specific reason in the farmer's decision-making pipeline:

### `/predict` — The Speed Layer (CNN)
**Why it exists:** Potato is the primary crop. Farmers need instant, reliable classification they can trust — even without internet for the Gemini API. The CNN model runs locally, responds in under a second, and costs nothing per request. It detects the two most common potato diseases (Early Blight, Late Blight) with high accuracy from a trained neural network.

**When to use it:** When the user knows they're looking at a potato leaf and wants the fastest possible answer.

### `/analyze/{plant_type}` — The Breadth Layer (Gemini Vision)
**Why it exists:** Training a dedicated CNN for every crop is impractical — it requires thousands of labeled images and weeks of training per plant. Instead, we use Gemini's vision capabilities guided by **expert prompts** that contain research-backed disease knowledge (disease names, visual symptoms, diagnostic criteria) sourced from university extension services. This lets us support 5 additional crops (tomato, corn, pepper, apple, strawberry) with 7-9 diseases each, without any model training.

**When to use it:** When the user has a non-potato plant, or wants a richer diagnosis with detailed descriptions of what the AI observes.

**Why two diagnosis paths?**
| | CNN (`/predict`) | Gemini (`/analyze`) |
|---|---|---|
| Speed | Sub-second | 2-5 seconds |
| Cost per request | Free (local model) | Gemini API cost |
| Offline capable | Yes | No |
| Plants supported | Potato only | Tomato, Corn, Pepper, Apple, Strawberry |
| Output detail | Class + confidence | Disease name + confidence + description |

### `/advice` — Turning Diagnosis into Action
**Why it exists:** A farmer who sees "Septoria Leaf Spot" on their screen needs to know **what to do**, not just what the disease is called. This endpoint generates plant-specific treatment recommendations, prevention strategies, and ongoing care instructions — so the user can act immediately after diagnosis.

**When to use it:** After any diagnosis (from `/predict` or `/analyze`) that identifies a disease. Pass `plant_type` to get advice tailored to the specific crop (e.g., fungicide recommendations differ between tomato and apple).

### `/crop-calendar` — Turning Advice into a Timeline
**Why it exists:** Knowing "apply fungicide" isn't enough — farmers need to know **when**, **in what order**, and **how often**. This endpoint generates a week-by-week action plan that factors in the current season and location, so the farmer can plan their work around weather patterns and crop growth stages.

**When to use it:** After getting advice, especially for moderate-to-severe cases that require multi-week treatment plans. The season and location fields help generate regionally appropriate timelines.

### `/severity` — Triage for Urgency
**Why it exists:** Not all infections need the same response speed. A mild early blight spot can wait until the weekend; a severe late blight outbreak needs action **today** before it destroys the entire field. This endpoint tells the user how bad the situation is (mild/moderate/severe), what percentage of the plant is affected, and what to do first.

**When to use it:** Alongside diagnosis, to prioritize response. Especially valuable when a farmer has multiple plants showing symptoms and needs to decide which to treat first.

### `/health` + `/ping` — Operational Monitoring
**Why they exist:** The frontend needs to show "service available" indicators, and deployment infrastructure (Koyeb, Docker health checks, load balancers) needs health-check probes. `/ping` is a simple liveness check; `/health` confirms the CNN model is actually loaded and ready to serve predictions.

---

## How Features Connect — The Decision Pipeline

```
User photographs a leaf
         |
         |--- Is it a potato?
         |         |
         |         +---> /predict (CNN — fast, free, offline)
         |         |        |
         |         |        +-- class: "Early Blight", confidence: 0.98
         |         |
         |         +-- Diseased? --+
         |                         |
         |                    +----+----+----+
         |                    |         |    |
         |                    v         v    v
         |               /advice   /calendar  /severity
         |              (what to   (when to   (how bad
         |               do)        do it)     is it)
         |
         |--- Is it tomato/corn/pepper/apple/strawberry?
                   |
                   +---> /analyze/{plant_type} (Gemini Vision)
                   |        |
                   |        +-- disease_name: "Septoria Leaf Spot"
                   |            confidence: 0.87
                   |            description: "Tiny circular spots..."
                   |
                   +-- Diseased? --+
                                   |
                              +----+----+
                              |         |
                              v         v
                         /advice   /calendar
                        (with      (with
                        plant_type) plant_type)
```

**Why two paths exist:**
- **CNN path** (`/predict`): Faster (~100ms), free (no API calls), works offline, but limited to 3 potato classes. Best for high-volume potato screening.
- **Gemini path** (`/analyze`): Slower (~3s), costs per request (Gemini API), needs internet, but covers 38+ diseases across 5 crops with rich descriptions. Best for multi-crop farms and detailed diagnosis.

Both paths feed into the same downstream endpoints (`/advice`, `/crop-calendar`) — the `plant_type` field ensures advice is tailored to the right crop.

---

## 1. GET `/ping`

Simple liveness check.

**Response:** `200 OK`
```
"Hello, I am alive"
```

---

## 2. GET `/health`

Returns API health status and whether the CNN model is loaded.

**Response:** `200 OK`
```json
{
  "status": "ok",
  "model_loaded": true
}
```

**Response Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `status` | `string` | Always `"ok"` |
| `model_loaded` | `boolean` | Whether the CNN model is loaded in memory |

---

## 3. POST `/predict`

Upload a potato leaf image to classify the disease.

**Rate limit:** 10 requests/minute per IP

### Request

- **Content-Type:** `multipart/form-data`
- **Field:** `file` (required) — the image file

**File constraints:**
- Max size: **10 MB**
- Allowed types: `image/jpeg`, `image/png`, `image/webp`

**Example (curl):**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@potato_leaf.jpg"
```

**Example (JavaScript fetch):**
```javascript
const formData = new FormData();
formData.append("file", fileInput.files[0]);

const res = await fetch("http://localhost:8000/predict", {
  method: "POST",
  body: formData,
});
const data = await res.json();
// data.class, data.confidence, data.confidence_percent
```

### Response: `200 OK`

```json
{
  "class": "Early Blight",
  "confidence": 0.9823,
  "confidence_percent": "98.2%"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `class` | `string` | One of: `"Early Blight"`, `"Late Blight"`, `"Healthy"` |
| `confidence` | `float` | Confidence score between 0 and 1 |
| `confidence_percent` | `string` | Human-readable confidence, e.g. `"98.2%"` |

---

## 4. POST `/advice`

Get AI-generated treatment advice for a detected disease.

**Rate limit:** 5 requests/minute per IP

### Request

- **Content-Type:** `application/json`

```json
{
  "disease": "Early Blight",
  "confidence": 0.9823
}
```

| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| `disease` | `string` | Yes | Disease name from `/predict` or `/analyze` |
| `confidence` | `float` | Yes | Between 0 and 1 (inclusive) |
| `plant_type` | `string` | No | Plant name (e.g. `"tomato"`, `"corn"`). Defaults to `"potato"` if omitted. Tailors advice to the specific crop. |

**Example (curl):**
```bash
curl -X POST http://localhost:8000/advice \
  -H "Content-Type: application/json" \
  -d '{"disease": "Early Blight", "confidence": 0.9823}'
```

**Example (JavaScript fetch):**
```javascript
const res = await fetch("http://localhost:8000/advice", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    disease: data.class,       // from /predict response
    confidence: data.confidence // from /predict response
  }),
});
const advice = await res.json();
```

### Response: `200 OK`

```json
{
  "disease": "Early Blight",
  "treatment": "Apply chlorothalonil or mancozeb-based fungicide every 7-10 days. Remove and destroy infected leaves immediately to prevent spore spread.",
  "prevention": "Practice crop rotation with non-solanaceous crops every 2-3 years. Ensure adequate spacing between plants for air circulation.",
  "care_instructions": "Water at the base of the plants to keep foliage dry. Monitor remaining leaves weekly for new lesion development."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `disease` | `string` | Echoed back from request |
| `treatment` | `string` | Treatment recommendations |
| `prevention` | `string` | Prevention tips |
| `care_instructions` | `string` | Ongoing care instructions |

---

## 5. POST `/crop-calendar`

Get an action timeline for managing a detected disease.

**Rate limit:** 5 requests/minute per IP

### Request

- **Content-Type:** `application/json`

```json
{
  "disease": "Late Blight",
  "confidence": 0.95,
  "season": "summer",
  "location": "Poland"
}
```

| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| `disease` | `string` | Yes | Disease name from `/predict` or `/analyze` |
| `confidence` | `float` | Yes | Between 0 and 1 (inclusive) |
| `season` | `string` | No | Defaults to `"current"` if omitted |
| `location` | `string` | No | Defaults to `"general"` if omitted |
| `plant_type` | `string` | No | Plant name (e.g. `"tomato"`). Defaults to `"potato"` if omitted. Tailors calendar to the specific crop. |

**Example (curl):**
```bash
curl -X POST http://localhost:8000/crop-calendar \
  -H "Content-Type: application/json" \
  -d '{"disease": "Late Blight", "confidence": 0.95, "season": "summer", "location": "Poland"}'
```

### Response: `200 OK`

```json
{
  "disease": "Late Blight",
  "season": "summer",
  "timeline": [
    {
      "week": "Week 1",
      "action": "Initial Treatment",
      "details": "Apply systemic fungicide (metalaxyl-based) immediately to all affected and neighboring plants."
    },
    {
      "week": "Week 2",
      "action": "Follow-up Spray",
      "details": "Apply contact fungicide (mancozeb) to protect new growth."
    },
    {
      "week": "Week 3-4",
      "action": "Monitoring",
      "details": "Inspect plants every 2-3 days for new lesions. Remove severely infected plants."
    },
    {
      "week": "Week 5-6",
      "action": "Preventive Care",
      "details": "Continue biweekly fungicide applications. Improve drainage around plants."
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `disease` | `string` | Echoed back from request |
| `season` | `string` | The season used (from request or default) |
| `timeline` | `ActionItem[]` | Array of 3-6 action items |

**ActionItem:**

| Field | Type | Description |
|-------|------|-------------|
| `week` | `string` | Time label, e.g. `"Week 1"`, `"Week 2-3"` |
| `action` | `string` | Short action title |
| `details` | `string` | Specific instructions |

---

## 6. POST `/severity`

Upload a potato leaf image for AI-powered severity analysis.

**Rate limit:** 5 requests/minute per IP

### Request

- **Content-Type:** `multipart/form-data`
- **Field:** `file` (required) — the image file

**File constraints:** same as `/predict` (10 MB max, JPEG/PNG/WebP)

**Example (curl):**
```bash
curl -X POST http://localhost:8000/severity \
  -F "file=@potato_leaf.jpg"
```

**Example (JavaScript fetch):**
```javascript
const formData = new FormData();
formData.append("file", fileInput.files[0]);

const res = await fetch("http://localhost:8000/severity", {
  method: "POST",
  body: formData,
});
const severity = await res.json();
```

### Response: `200 OK`

```json
{
  "severity_level": "moderate",
  "affected_area_percent": 35.0,
  "urgency": "high",
  "immediate_actions": [
    "Isolate affected plants from healthy ones",
    "Apply appropriate fungicide within 24 hours",
    "Remove severely damaged leaves"
  ],
  "detailed_analysis": "The leaf shows moderate signs of fungal infection with brown lesions covering approximately 35% of the surface. The concentric ring pattern is characteristic of early blight. Immediate treatment is recommended to prevent spread to adjacent plants."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `severity_level` | `string` | `"mild"`, `"moderate"`, or `"severe"` |
| `affected_area_percent` | `float` | 0-100, estimated affected leaf area |
| `urgency` | `string` | `"low"`, `"medium"`, `"high"`, or `"critical"` |
| `immediate_actions` | `string[]` | 2-4 recommended immediate actions |
| `detailed_analysis` | `string` | 2-3 sentence analysis of the image |

---

## Suggested Frontend User Flow

The frontend should offer **two paths** based on what plant the user is examining:

```
┌─────────────────┐
│  User uploads    │
│  leaf image      │
└────────┬────────┘
         │
         ├── User selects "Potato"
         │         │
         │    ┌────▼─────┐     ┌──────────┐
         │    │ /predict  │────>│/severity │  (parallel)
         │    │ (CNN)     │     └──────────┘
         │    └────┬──────┘
         │         │ class + confidence
         │         │
         │    Diseased? ──Yes──> /advice + /crop-calendar
         │                       (plant_type omitted = potato)
         │
         └── User selects "Tomato" / "Corn" / "Pepper" / "Apple" / "Strawberry"
                   │
              ┌────▼──────────────┐
              │ /analyze/{plant}   │
              │ (Gemini Vision)    │
              └────┬──────────────┘
                   │ disease_name + confidence + description
                   │
              Diseased? ──Yes──> /advice + /crop-calendar
                                 (with plant_type for crop-specific guidance)
```

### Path A — Potato flow (CNN, fast, free):

```javascript
const formData = new FormData();
formData.append("file", imageFile);

// Step 1: Classify with CNN (fast, no API cost)
const prediction = await fetch("/predict", { method: "POST", body: formData }).then(r => r.json());
// { class: "Early Blight", confidence: 0.98, confidence_percent: "98.0%" }

// Step 2: Severity analysis (run in parallel with steps 3-4)
const severityForm = new FormData();
severityForm.append("file", imageFile);
const severity = await fetch("/severity", { method: "POST", body: severityForm }).then(r => r.json());

// Step 3: If diseased, get treatment advice (plant_type omitted = defaults to potato)
if (prediction.class !== "Healthy") {
  const advice = await fetch("/advice", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ disease: prediction.class, confidence: prediction.confidence }),
  }).then(r => r.json());

  // Step 4: Get action timeline
  const calendar = await fetch("/crop-calendar", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      disease: prediction.class,
      confidence: prediction.confidence,
      season: "summer",
      location: "Poland",
    }),
  }).then(r => r.json());
}
```

### Path B — Multi-plant flow (Gemini Vision):

```javascript
const formData = new FormData();
formData.append("file", imageFile);
const plantType = "tomato"; // user-selected

// Step 1: Analyze with Gemini (expert plant-specific prompt)
const analysis = await fetch(`/analyze/${plantType}`, { method: "POST", body: formData }).then(r => r.json());
// { plant_type: "tomato", disease_name: "Septoria Leaf Spot", is_healthy: false,
//   confidence: 0.87, description: "Tiny circular spots with gray-tan centers..." }

// Step 2: If diseased, get plant-specific advice
if (!analysis.is_healthy) {
  const [advice, calendar] = await Promise.all([
    fetch("/advice", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        disease: analysis.disease_name,
        confidence: analysis.confidence,
        plant_type: analysis.plant_type,  // "tomato" — tailors advice to tomato
      }),
    }).then(r => r.json()),

    fetch("/crop-calendar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        disease: analysis.disease_name,
        confidence: analysis.confidence,
        plant_type: analysis.plant_type,
        season: "summer",
        location: "Poland",
      }),
    }).then(r => r.json()),
  ]);
}
```

---

## File Upload Constraints (Client-Side Validation)

Enforce these before sending requests to avoid 413/415 errors:

| Constraint | Value |
|------------|-------|
| Max file size | 10 MB |
| Allowed MIME types | `image/jpeg`, `image/png`, `image/webp` |

```javascript
function validateFile(file) {
  const MAX_SIZE = 10 * 1024 * 1024; // 10 MB
  const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];

  if (!ALLOWED_TYPES.includes(file.type)) {
    return { valid: false, error: `File type "${file.type}" not allowed. Use JPEG, PNG, or WebP.` };
  }
  if (file.size > MAX_SIZE) {
    return { valid: false, error: "File exceeds 10 MB limit." };
  }
  return { valid: true };
}
```

---

## Error Handling Guide

All errors return JSON with a `detail` field. Some include an `error_code`.

```json
{
  "detail": "Human-readable error message",
  "error_code": "OPTIONAL_CODE"
}
```

### Error Codes Reference

| Status | Cause | `detail` example | When it happens |
|--------|-------|-------------------|-----------------|
| **400** | Bad request | `"Could not read image file. It may be corrupt."` | Uploaded file is not a valid image |
| **413** | File too large | `"File too large. Maximum size: 10MB"` | Image exceeds 10 MB |
| **415** | Wrong file type | `"Unsupported file type 'text/plain'. Allowed: [...]"` | Non-image file uploaded |
| **422** | Validation error | `[{"loc": ["body","confidence"], "msg": "..."}]` | JSON body fails Pydantic validation (e.g., confidence > 1) |
| **429** | Rate limited | `"Rate limit exceeded"` | Too many requests per minute |
| **500** | Server error | `"Prediction failed"` / `"Internal server error"` | CNN model error or unhandled exception |
| **502** | AI service error | `"AI advice generation failed"` / `"AI calendar generation failed"` / `"AI severity analysis failed"` | Gemini API error |

### Handling errors in the frontend

```javascript
async function apiCall(url, options) {
  const res = await fetch(url, options);

  if (!res.ok) {
    const error = await res.json();

    switch (res.status) {
      case 413:
      case 415:
        // Show file validation message to user
        showError(error.detail);
        break;
      case 422:
        // Show form validation error
        showError("Invalid input. Please check your values.");
        break;
      case 429:
        // Rate limited - show retry message
        showError("Too many requests. Please wait a moment and try again.");
        break;
      case 502:
        // AI service down - suggest retry
        showError("AI service temporarily unavailable. Please try again.");
        break;
      default:
        showError("Something went wrong. Please try again later.");
    }
    return null;
  }

  return res.json();
}
```

---

## 7. POST `/analyze/{plant_type}`

Analyze a plant leaf image for disease using Gemini vision with expert plant-specific prompts.

**Rate limit:** 5 requests/minute per IP

**Supported plant types:** `tomato`, `corn`, `pepper`, `apple`, `strawberry`

### Request

- **Content-Type:** `multipart/form-data`
- **Path parameter:** `plant_type` (required) — one of the supported plant types
- **Field:** `file` (required) — the image file

**File constraints:** same as `/predict` (10 MB max, JPEG/PNG/WebP)

**Example (curl):**
```bash
curl -X POST http://localhost:8000/analyze/tomato \
  -F "file=@tomato_leaf.jpg"
```

**Example (JavaScript fetch):**
```javascript
const formData = new FormData();
formData.append("file", fileInput.files[0]);

const res = await fetch("http://localhost:8000/analyze/tomato", {
  method: "POST",
  body: formData,
});
const analysis = await res.json();
```

### Response: `200 OK`

```json
{
  "plant_type": "tomato",
  "disease_name": "Septoria Leaf Spot",
  "is_healthy": false,
  "confidence": 0.87,
  "description": "The leaf shows numerous tiny circular spots with gray-tan centers and visible dark pycnidia. The spots are concentrated on lower leaves, which is characteristic of Septoria leaf spot."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `plant_type` | `string` | The plant type from the request path |
| `disease_name` | `string` | Identified disease or `"Healthy"` or `"Not Identified"` |
| `is_healthy` | `boolean` | `true` only if the plant appears healthy |
| `confidence` | `float` | Confidence score between 0 and 1 |
| `description` | `string` | 2-3 sentence description of observations |

### Using analyze results with `/advice` and `/crop-calendar`

The analyze result can feed directly into the advice and crop calendar endpoints by passing the `plant_type` field:

```javascript
// Step 1: Analyze plant image
const analysis = await fetch("/analyze/tomato", { method: "POST", body: formData }).then(r => r.json());

// Step 2: Get plant-specific advice
if (!analysis.is_healthy) {
  const advice = await fetch("/advice", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      disease: analysis.disease_name,
      confidence: analysis.confidence,
      plant_type: analysis.plant_type,  // makes advice tomato-specific
    }),
  }).then(r => r.json());

  // Step 3: Get plant-specific crop calendar
  const calendar = await fetch("/crop-calendar", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      disease: analysis.disease_name,
      confidence: analysis.confidence,
      plant_type: analysis.plant_type,
    }),
  }).then(r => r.json());
}
```

### Error: Invalid plant type → `422`

```bash
curl -X POST http://localhost:8000/analyze/banana -F "file=@leaf.jpg"
# → 422: value is not a valid enumeration member
```

---

## CORS

The API allows all origins by default (`*`). No special headers or credentials configuration needed from the frontend.
