# Architecture Visualization — Plant Disease Classifier API

## Context
The multi-plant disease analysis feature has been implemented. This document visualizes the complete system architecture, every user flow path, and how data moves through each layer.

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React + MUI)                          │
│  potato-disease-frontend/src/home.js                                    │
│                                                                         │
│  ┌─────────────┐  ┌───────────────┐  ┌──────────┐  ┌────────────────┐  │
│  │  DropzoneArea│  │ Image Preview │  │ Results  │  │ Clear Button   │  │
│  │  (upload)    │  │ (CardMedia)   │  │ (Table)  │  │                │  │
│  └──────┬───────┘  └───────────────┘  └──────────┘  └────────────────┘  │
│         │                                                               │
│         │  FormData { file }                                            │
│         ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  axios.post(API_URL + endpoint, formData / jsonBody)            │    │
│  └──────────────────────────────┬──────────────────────────────────┘    │
└─────────────────────────────────┼───────────────────────────────────────┘
                                  │ HTTP
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    FASTAPI APPLICATION (main.py)                        │
│                    Plant Disease Classifier API v3.0                    │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  MIDDLEWARE LAYER                                                 │   │
│  │  ├─ CORSMiddleware (allow_origins=["*"])                        │   │
│  │  ├─ slowapi.Limiter (per-IP rate limiting)                      │   │
│  │  └─ Global exception handler → 500 JSON                        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  ROUTER LAYER                                                    │   │
│  │                                                                  │   │
│  │  health.router ─── GET /ping, GET /health                       │   │
│  │  predict.router ── POST /predict          [10/min]              │   │
│  │  analyze.router ── POST /analyze/{plant}  [5/min]  ◄── NEW     │   │
│  │  advice.router ─── POST /advice           [5/min]  ◄── UPDATED │   │
│  │  calendar.router ─ POST /crop-calendar    [5/min]  ◄── UPDATED │   │
│  │  severity.router ─ POST /severity         [5/min]               │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  SERVICE LAYER                                                   │   │
│  │                                                                  │   │
│  │  image_service.py                                                │   │
│  │  └─ validate_and_read_image() → (np.ndarray, bytes)             │   │
│  │                                                                  │   │
│  │  gemini_service.py                                               │   │
│  │  ├─ generate_text(prompt, model) → str                          │   │
│  │  └─ analyze_image(bytes, mime, prompt, model) → str             │   │
│  │                                                                  │   │
│  │  model_loader.py                                                 │   │
│  │  └─ get_model() → tf.keras.Model                                │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  KNOWLEDGE LAYER  (NEW)                                          │   │
│  │                                                                  │   │
│  │  prompts/plant_prompts.py                                        │   │
│  │  ├─ TOMATO_PROMPT  (9 diseases)                                 │   │
│  │  ├─ CORN_PROMPT    (8 diseases)                                 │   │
│  │  ├─ PEPPER_PROMPT  (7 diseases)                                 │   │
│  │  ├─ APPLE_PROMPT   (7 diseases)                                 │   │
│  │  ├─ STRAWBERRY_PROMPT (7 diseases)                              │   │
│  │  └─ get_plant_prompt(plant_type) → str                          │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└───────────────────┬─────────────────────┬───────────────────────────────┘
                    │                     │
                    ▼                     ▼
         ┌──────────────────┐  ┌─────────────────────┐
         │  TensorFlow/Keras│  │  Google Gemini API   │
         │  CNN Model       │  │  (gemini-2.5-flash)  │
         │  cnn_model.keras │  │                      │
         │                  │  │  ├─ Vision analysis  │
         │  3 classes:      │  │  └─ Text generation  │
         │  Early Blight    │  │                      │
         │  Late Blight     │  │  Auth: API key from  │
         │  Healthy         │  │  POTATO_GEMINI_API_KEY│
         └──────────────────┘  └─────────────────────┘
```

---

## 2. User Flow A — Potato Disease (CNN Path, Original)

```
    USER                      FRONTEND                    BACKEND
     │                           │                           │
     │  Drops potato leaf image  │                           │
     │ ─────────────────────────>│                           │
     │                           │                           │
     │                           │  POST /predict            │
     │                           │  Content-Type: multipart  │
     │                           │  Body: { file: image }    │
     │                           │ ─────────────────────────>│
     │                           │                           │
     │                           │                     ┌─────┴──────┐
     │                           │                     │ ROUTE:     │
     │                           │                     │ predict.py │
     │                           │                     └─────┬──────┘
     │                           │                           │
     │                           │              validate_and_read_image()
     │                           │                     ┌─────┴──────┐
     │                           │                     │ Check MIME │
     │                           │                     │ Check size │
     │                           │                     │ PIL→numpy  │
     │                           │                     └─────┬──────┘
     │                           │                           │
     │                           │                    get_model().predict()
     │                           │                     ┌─────┴──────┐
     │                           │                     │ CNN Model  │
     │                           │                     │ 256×256×3  │
     │                           │                     │ → softmax  │
     │                           │                     │ [0.02,     │
     │                           │                     │  0.96,     │
     │                           │                     │  0.02]     │
     │                           │                     └─────┬──────┘
     │                           │                           │
     │                           │  { "class": "Late Blight",│
     │                           │    "confidence": 0.96,    │
     │                           │    "confidence_percent":  │
     │                           │        "96.0%" }          │
     │                           │ <─────────────────────────│
     │                           │                           │
     │  Shows: Late Blight 96%   │                           │
     │ <─────────────────────────│                           │
```

---

## 3. User Flow B — Multi-Plant Analysis (Gemini Vision Path, NEW)

```
    USER                      FRONTEND                    BACKEND
     │                           │                           │
     │  Selects "Tomato"         │                           │
     │  Drops tomato leaf image  │                           │
     │ ─────────────────────────>│                           │
     │                           │                           │
     │                           │  POST /analyze/tomato     │
     │                           │  Body: { file: image }    │
     │                           │ ─────────────────────────>│
     │                           │                           │
     │                           │                     ┌─────┴──────────┐
     │                           │                     │ ROUTE:         │
     │                           │                     │ analyze.py     │
     │                           │                     └─────┬──────────┘
     │                           │                           │
     │                           │              validate_and_read_image()
     │                           │                           │
     │                           │              get_plant_prompt("tomato")
     │                           │                     ┌─────┴──────────┐
     │                           │                     │ TOMATO_PROMPT  │
     │                           │                     │ 9 diseases +   │
     │                           │                     │ visual IDs +   │
     │                           │                     │ healthy ref +  │
     │                           │                     │ JSON format    │
     │                           │                     └─────┬──────────┘
     │                           │                           │
     │                           │              analyze_image(bytes, mime,
     │                           │                  prompt, "gemini-2.5-flash")
     │                           │                     ┌─────┴──────────┐
     │                           │                     │ GEMINI API     │
     │                           │                     │ Vision call    │
     │                           │                     │                │
     │                           │                     │ Receives:      │
     │                           │                     │ - Expert prompt│
     │                           │                     │ - Leaf image   │
     │                           │                     │                │
     │                           │                     │ Returns JSON:  │
     │                           │                     │ disease_name,  │
     │                           │                     │ is_healthy,    │
     │                           │                     │ confidence,    │
     │                           │                     │ description    │
     │                           │                     └─────┬──────────┘
     │                           │                           │
     │                           │              Parse JSON (strip markdown
     │                           │              if present, extract {})
     │                           │                           │
     │                           │  { "plant_type": "tomato",│
     │                           │    "disease_name":        │
     │                           │       "Septoria Leaf Spot",│
     │                           │    "is_healthy": false,   │
     │                           │    "confidence": 0.87,    │
     │                           │    "description": "..." } │
     │                           │ <─────────────────────────│
     │                           │                           │
     │  Shows diagnosis result   │                           │
     │ <─────────────────────────│                           │
```

---

## 4. User Flow C — Full Pipeline: Analyze → Advice → Calendar

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │  STEP 1: DIAGNOSE                                                    │
 │                                                                      │
 │  POST /analyze/strawberry                                            │
 │  Body: multipart { file: strawberry_leaf.jpg }                       │
 │                                                                      │
 │  ┌─ analyze.py ─┐    ┌─ plant_prompts.py ─┐    ┌─ Gemini Vision ─┐ │
 │  │ validate img  │───>│ STRAWBERRY_PROMPT  │───>│ image + prompt  │ │
 │  │ get prompt    │    │ 7 diseases         │    │ → JSON response │ │
 │  │ call Gemini   │    └────────────────────┘    └────────┬────────┘ │
 │  │ parse JSON    │<──────────────────────────────────────┘          │
 │  └───────┬───────┘                                                  │
 │          │                                                          │
 │          ▼                                                          │
 │  Response: {                                                        │
 │    plant_type: "strawberry",                                        │
 │    disease_name: "Gray Mold (Botrytis)",                            │
 │    is_healthy: false,                                               │
 │    confidence: 0.91,                                                │
 │    description: "Gray fuzzy growth covering fruit..."               │
 │  }                                                                  │
 └──────────────────────────────┬───────────────────────────────────────┘
                                │
                    Frontend extracts disease_name,
                    confidence, plant_type from response
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
 ┌──────────────────────────────┐  ┌───────────────────────────────────┐
 │  STEP 2: GET ADVICE          │  │  STEP 3: GET CALENDAR             │
 │                              │  │  (can run in parallel)            │
 │  POST /advice                │  │  POST /crop-calendar              │
 │  {                           │  │  {                                │
 │    disease: "Gray Mold       │  │    disease: "Gray Mold            │
 │             (Botrytis)",     │  │             (Botrytis)",          │
 │    confidence: 0.91,         │  │    confidence: 0.91,              │
 │    plant_type: "strawberry"  │  │    plant_type: "strawberry",     │
 │  }                           │  │    season: "spring",              │
 │                              │  │    location: "Poland"             │
 │  ┌─ advice.py ──────────┐   │  │  }                                │
 │  │ Build prompt:         │   │  │                                   │
 │  │ "advisor specializing │   │  │  ┌─ calendar.py ──────────┐      │
 │  │  in STRAWBERRY        │   │  │  │ Build prompt:           │      │
 │  │  diseases"            │   │  │  │ "planner for STRAWBERRY │      │
 │  │                       │   │  │  │  crops"                 │      │
 │  │ generate_text()       │   │  │  │                         │      │
 │  │  → Gemini Text API    │   │  │  │ generate_text()         │      │
 │  │  → parse sections     │   │  │  │  → Gemini Text API      │      │
 │  └───────────┬───────────┘   │  │  │  → parse JSON array     │      │
 │              ▼               │  │  └───────────┬─────────────┘      │
 │  {                           │  │              ▼                    │
 │    disease: "Gray Mold...",  │  │  {                                │
 │    treatment: "Remove        │  │    disease: "Gray Mold...",       │
 │      infected fruit...",     │  │    season: "spring",              │
 │    prevention: "Improve      │  │    timeline: [                    │
 │      air circulation...",    │  │      { week: "Week 1",            │
 │    care_instructions:        │  │        action: "Sanitation",      │
 │      "Monitor humidity..."   │  │        details: "Remove all       │
 │  }                           │  │          infected fruit..." },    │
 └──────────────────────────────┘  │      { week: "Week 2-3", ... },  │
                                   │      { week: "Week 4-6", ... }   │
                                   │    ]                              │
                                   │  }                                │
                                   └───────────────────────────────────┘
```

---

## 5. User Flow D — Severity Analysis (Image-Based, Independent)

```
     POST /severity
     Body: multipart { file: leaf_image.jpg }
              │
              ▼
     ┌─ severity.py ────────────────────────────┐
     │                                           │
     │  validate_and_read_image()                │
     │       │                                   │
     │       ▼                                   │
     │  Hardcoded severity prompt:               │
     │  "Analyze this potato leaf image          │
     │   for disease severity..."                │
     │       │                                   │
     │       ▼                                   │
     │  analyze_image() → Gemini Vision API      │
     │       │                                   │
     │       ▼                                   │
     │  Parse JSON response                      │
     │  (fallback if unparseable)                │
     └───────┬───────────────────────────────────┘
             ▼
     {
       severity_level: "moderate",
       affected_area_percent: 35.0,
       urgency: "high",
       immediate_actions: [
         "Isolate affected plants",
         "Apply fungicide within 24h",
         "Remove damaged leaves"
       ],
       detailed_analysis: "Brown lesions covering ~35%..."
     }
```

---

## 6. Data Model Relationships

```
     ┌─────────────────┐         ┌──────────────────────┐
     │   PlantType      │         │  PredictionResponse   │
     │   (Enum)         │         │  (CNN output)         │
     │                  │         │                       │
     │  tomato          │         │  class: str           │
     │  corn            │         │  confidence: float    │
     │  pepper          │         │  confidence_percent   │
     │  apple           │         └──────────────────────┘
     │  strawberry      │
     └────────┬─────────┘
              │ validates path param
              ▼
     ┌──────────────────────┐
     │  AnalyzeResponse      │
     │  (Gemini output)      │
     │                       │       ┌─────────────────────┐
     │  plant_type: str ─────────────│ Feeds into:         │
     │  disease_name: str ───────────│                     │
     │  is_healthy: bool     │       │  AdviceRequest      │
     │  confidence: float ───────────│  ├ disease: str     │
     │  description: str     │       │  ├ confidence: float│
     └──────────────────────┘       │  └ plant_type?: str │
                                     │                     │
                                     │  CropCalendarRequest│
                                     │  ├ disease: str     │
                                     │  ├ confidence: float│
                                     │  ├ plant_type?: str │
                                     │  ├ season?: str     │
                                     │  └ location?: str   │
                                     └─────────────────────┘
                                              │
                              ┌───────────────┴──────────────┐
                              ▼                              ▼
                    ┌──────────────────┐         ┌────────────────────┐
                    │ AdviceResponse    │         │CropCalendarResponse│
                    │                  │         │                    │
                    │ disease: str     │         │ disease: str       │
                    │ treatment: str   │         │ season: str        │
                    │ prevention: str  │         │ timeline:          │
                    │ care_instructions│         │  [ActionItem]      │
                    └──────────────────┘         │   ├ week: str      │
                                                 │   ├ action: str    │
                                                 │   └ details: str   │
                                                 └────────────────────┘
```

---

## 7. Prompt Selection Flow (Knowledge Layer)

```
     POST /analyze/{plant_type}
              │
              │  plant_type = "corn"
              ▼
     get_plant_prompt("corn")
              │
              ▼
     ┌─ _PROMPTS dictionary ──────────────────────────────┐
     │                                                     │
     │  "tomato"     → TOMATO_PROMPT     (2083 chars)     │
     │  "corn"       → CORN_PROMPT       (1861 chars)  ◄──│── selected
     │  "pepper"     → PEPPER_PROMPT     (1756 chars)     │
     │  "apple"      → APPLE_PROMPT      (1843 chars)     │
     │  "strawberry" → STRAWBERRY_PROMPT (1846 chars)     │
     └─────────────────────────┬───────────────────────────┘
                               │
                               ▼
     ┌─ CORN_PROMPT structure ────────────────────────────┐
     │                                                     │
     │  "You are an expert plant pathologist               │
     │   specializing in corn (maize) diseases."           │
     │                                                     │
     │  Disease Table:                                     │
     │  ┌──────────────────┬──────────────────────────┐   │
     │  │ Common Rust      │ cinnamon-brown pustules   │   │
     │  │ Gray Leaf Spot   │ rectangular tan-gray      │   │
     │  │ N. Leaf Blight   │ cigar-shaped 1-6"         │   │
     │  │ S. Leaf Blight   │ diamond-shaped small      │   │
     │  │ Common Smut      │ tumor-like galls          │   │
     │  │ Dwarf Mosaic     │ yellow-green streaks      │   │
     │  │ Stewart's Wilt   │ wavy yellow-grey streaks  │   │
     │  │ Anthracnose      │ reddish-brown borders     │   │
     │  └──────────────────┴──────────────────────────┘   │
     │                                                     │
     │  Healthy Reference: uniformly green, no spots       │
     │                                                     │
     │  Task: verify plant → diagnose → JSON output        │
     │                                                     │
     │  Required JSON:                                     │
     │  { disease_name, is_healthy, confidence,            │
     │    description }                                    │
     └─────────────────────────────────────────────────────┘
```

---

## 8. Error Handling Flow

```
     Incoming Request
          │
          ▼
     ┌─ PlantType enum validation ─────────────────────┐
     │  /analyze/banana → 422 Validation Error          │
     │  (automatic from FastAPI + Enum)                 │
     └──────────────────────────┬───────────────────────┘
                                │ valid plant_type
                                ▼
     ┌─ validate_and_read_image() ─────────────────────┐
     │  Wrong MIME type → 415 Unsupported Media Type    │
     │  File > 10MB    → 413 Payload Too Large          │
     │  Corrupt image  → 400 Bad Request                │
     └──────────────────────────┬───────────────────────┘
                                │ valid image
                                ▼
     ┌─ Gemini API call ───────────────────────────────┐
     │  No API key / client not init → 503 Unavailable  │
     │  Gemini API error             → 502 Bad Gateway  │
     └──────────────────────────┬───────────────────────┘
                                │ got response text
                                ▼
     ┌─ JSON parsing ──────────────────────────────────┐
     │  Valid JSON     → AnalyzeResponse with data      │
     │  Invalid JSON   → Fallback AnalyzeResponse:      │
     │                   disease_name: "Not Identified"  │
     │                   confidence: 0.0                 │
     │                   description: raw text[:300]     │
     └─────────────────────────────────────────────────┘
```

---

## 9. Backward Compatibility — Existing Potato Flow Still Works

```
     EXISTING FLOW (unchanged):

     POST /predict { file: potato.jpg }
          │
          ▼  CNN model → { class: "Early Blight", confidence: 0.98 }
          │
          ▼
     POST /advice { disease: "Early Blight", confidence: 0.98 }
          │        (no plant_type field)
          │
          ▼  plant = body.plant_type or "potato"  →  "potato"
          │  Prompt: "advisor specializing in potato diseases"
          │
          ▼  Works exactly as before


     NEW FLOW (added):

     POST /analyze/tomato { file: tomato.jpg }
          │
          ▼  Gemini → { disease_name: "Late Blight", confidence: 0.85 }
          │
          ▼
     POST /advice { disease: "Late Blight", confidence: 0.85,
                    plant_type: "tomato" }
          │
          ▼  plant = body.plant_type or "potato"  →  "tomato"
          │  Prompt: "advisor specializing in tomato diseases"
```

---

## 10. File Map — What Lives Where

```
potato-disease-classifier/
├── main.py                      ← App init, routers, middleware, lifespan
├── config.py                    ← Settings (env vars, rate limits, model path)
├── schemas.py                   ← PlantType enum + all request/response models
├── model_loader.py              ← TensorFlow CNN model loading
│
├── prompts/                     ← NEW: Expert knowledge layer
│   ├── __init__.py
│   └── plant_prompts.py         ← 5 plant prompts + get_plant_prompt()
│
├── routes/
│   ├── health.py                ← GET /ping, /health
│   ├── predict.py               ← POST /predict (CNN)
│   ├── analyze.py               ← NEW: POST /analyze/{plant_type} (Gemini)
│   ├── advice.py                ← POST /advice (UPDATED: plant-agnostic)
│   ├── calendar.py              ← POST /crop-calendar (UPDATED: plant-agnostic)
│   └── severity.py              ← POST /severity
│
├── services/
│   ├── gemini_service.py        ← Gemini client: generate_text, analyze_image
│   └── image_service.py         ← validate_and_read_image()
│
├── cnn_model.keras              ← Trained CNN weights
└── README.md                    ← UPDATED: documents /analyze endpoint
```
