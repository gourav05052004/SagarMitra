# SagarMitra (सागरमित्र) 🌊🚁
### Autonomous Deep-Learning Maritime Surveillance & Search-and-Rescue (SAR) System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0.1+cu118](https://img.shields.io/badge/PyTorch-2.0.1%2Bcu118-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MMDetection 2.28.2](https://img.shields.io/badge/MMDetection-2.28.2-blue?logo=openmmlab&logoColor=white)](https://github.com/open-mmlab/mmdetection)
[![Ultralytics YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11s-00FFFF?logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0.0-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![AWS EC2 & S3](https://img.shields.io/badge/AWS-EC2%20%7C%20S3%20%7C%20SSM-232F3E?logo=amazon-aws&logoColor=white)](https://aws.amazon.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Flutter 3.x](https://img.shields.io/badge/Flutter-Cross--Platform-02569B?logo=flutter&logoColor=white)](https://flutter.dev/)
[![Benchmark Score](https://img.shields.io/badge/MaCVi%20AP50%3A95-61.57%25-brightgreen)](#-empirical-benchmarks--research-results)

> **SagarMitra** (*Friend of the Ocean*) is an end-to-end, multi-stage computer vision and cloud-edge platform engineered for high-precision autonomous maritime search-and-rescue (SAR), coastal surveillance, and small-object detection in harsh open-water environments.

---

## 📌 Table of Contents
1. [Executive Summary & Problem Statement](#-executive-summary--problem-statement)
2. [Empirical Benchmarks & Research Results](#-empirical-benchmarks--research-results)
3. [Algorithmic Architecture (4-Stage Staged Pipeline)](#-algorithmic-architecture-4-stage-staged-pipeline)
4. [End-to-End System & Cloud Architecture](#-end-to-end-system--cloud-architecture)
5. [Client Application (Flutter Material 3)](#-client-application-flutter-material-3)
6. [API Specification & Data Telemetry](#-api-specification--data-telemetry)
7. [Repository Structure](#-repository-structure)
8. [Setup, Deployment & Reproduction Guide](#-setup-deployment--reproduction-guide)


---

## 🌊 Executive Summary & Problem Statement

Maritime computer vision is plagued by extreme environmental degradation that causes conventional object detectors (e.g., standard single-stage YOLO or standalone Faster R-CNN) to fail catastrophically:
- **Severe Scale Disparity:** Critical life-saving targets (swimmers, life-jackets, buoys) typically occupy fewer than **$16 \times 16$ pixels** across ultra-high-resolution aerial or coastal camera feeds.
- **Dynamic Optical Interference:** Sun glint, breaking wave crests, specular water reflections, and foam trails create persistent high-frequency false positives.
- **Camouflage & Occlusion:** Swimmers in choppy water are intermittently submerged and occluded by swell peaks.

**SagarMitra** overcomes these fundamental challenges through a **4-stage progressive ensemble pipeline**:
1. Combining heavy composite hierarchical vision transformers (**CBNetV2 + Swin-Small**) for dense context and multi-scale feature extraction.
2. Refining spatial boundaries and class distributions with **edge-aligned residual heads and geometric tree normalization**.
3. Fusing high-speed regression-detail features from an expert **YOLO11s** network.
4. Executing **specialized tiled micro-inference** focused on sub-16px maritime targets.

The entire system is operationalized via a **Dockerized FastAPI microservice on AWS EC2** backed by an **Elastic IP**, persistent **SQLite research database**, **Amazon S3** artifact storage, and a cross-platform **Flutter mobile app**.

---

## 📊 Empirical Benchmarks & Research Results

Evaluated rigorously on **1,547 unseen validation images** under the competitive **MaCVi / SeaDronesSee Maritime Benchmark**, the SagarMitra staged pipeline achieves state-of-the-art detection precision:

| Metric | SagarMitra Staged Pipeline | Standard Standalone Detector | Absolute Gain |
| :--- | :---: | :---: | :---: |
| **mAP (COCO bbox $AP_{50:95}$)** | **61.57%** | ~48.20% | **+13.37%** |
| **$AP_{50}$ (PASCAL VOC Metric)** | **91.53%** | ~78.40% | **+13.13%** |
| **$AP_{75}$ (Strict IoU Threshold)** | **68.78%** | ~52.10% | **+16.68%** |
| **$AP_{small}$ (Targets $< 32^2$ px)** | **57.20%** | ~34.80% | **+22.40%** |
| **$AR_{100}$ (Average Recall @ 100)** | **74.77%** | ~61.30% | **+13.47%** |

> [!IMPORTANT]
> The **$57.20\% \ AP_{small}$** metric is the cornerstone of this research. In maritime search-and-rescue, detecting a drowning swimmer or floating life-vest at vast standoff distances saves human lives. Standalone models frequently miss these targets, whereas SagarMitra's tiled micro-correction recovers them with high statistical confidence.

### Target Classification Ontology (5 Marine Classes)
| Class ID | Target Category | Operational Priority | Characteristic Challenges |
| :---: | :--- | :---: | :--- |
| **1** | **Swimmer** | 🚨 Emergency (Critical) | Extremely small signature ($<16$ px), head-only visibility, water swell occlusion. |
| **2** | **Boat** | 🚢 Navigation / Defense | Large scale variability (skiffs to commercial vessels), wake distortion. |
| **3** | **Jetski** | ⚡ High-Speed Craft | Rapid motion blur, spray occlusion, non-standard orientation. |
| **4** | **Life Saving Appliance (LSA)** | 🚨 Emergency (Critical) | High-contrast color (orange/yellow) but tiny geometric footprint in open water. |
| **5** | **Buoy** | ⚓ Navigation Aid | Wave-bobbing oscillation, mistook for floating debris or whitecap foam. |

---

## 🧠 Algorithmic Architecture (4-Stage Staged Pipeline)

The system does not rely on a single monolithic checkpoint. High benchmark performance is produced through a sequential, multi-scale refinement sequence:

```
                          Input Maritime Image (1920x1280)
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             ▼                                                     ▼
     Full-Image Inference                               1536x1536 Sliding Tiles
  (CBNetV2 Swin-S Cascade)                                  (25% Overlap)
             │                                                     │
             └──────────────────────────┬──────────────────────────┘
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │   STAGE 1: Primary Composite CBNetV2 + Swin-S       │
             │   - Dual-Backbone composite architecture            │
             │   - Cascade R-CNN multi-stage box regressors        │
             └──────────────────────────┬──────────────────────────┘
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │   STAGE 2: Retained Edge & Geometric Refinement     │
             │   - Edge-aligned bounding-box residual head         │
             │   - Class-safe confidence distribution calibration  │
             │   - Learning-to-rank scoring normalization          │
             │   - Hierarchical geometric decision tree            │
             │   - Pixel-boundary alignment network                │
             └──────────────────────────┬──────────────────────────┘
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │   STAGE 3: YOLO11s Complementary Expert Fusion      │
             │   - Full-image 1280px regression-detail detector    │
             │   - Weighted IoU association (IoU >= 0.55)          │
             │   - Box weight: 0.40 | Score weight: 0.15           │
             └──────────────────────────┬──────────────────────────┘
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │   STAGE 4: Tiled Micro-Correction (Sub-16px SAR)    │
             │   - Dedicated sub-window inference for classes 1 & 4│
             │   - Soft-NMS suppression for dense clusters         │
             └──────────────────────────┬──────────────────────────┘
                                        ▼
                           Final Filtered Detections
                      (Confidence >= 0.50 + Coordinates)
```

1. **Stage 1 (Primary Dual-Backbone Detector):** Utilizes composite connections between Swin-Small Transformer backbones (CBNetV2 architecture) feeding a Cascade R-CNN head. Operates over both full-scale inputs and $1536 \times 1536$ sliding tiles with $25\%$ overlap.
2. **Stage 2 (Retained Geometric & Boundary Refiners):** Corrects spatial bounding-box edge jitter caused by water glare and calibrates class probabilities using frozen ranking policies and a geometric decision tree.
3. **Stage 3 (Complementary Expert Fusion):** Integrates YOLO11s regression detail outputs to resolve ambiguities in dense boat/jetski formations via weighted box averaging.
4. **Stage 4 (Tiled Expert Correction for Tiny Objects):** Activates specialized slicing for bounding boxes whose effective diagonal size is $<16$ pixels, explicitly enhancing recall for Swimmers and Life Saving Appliances.

---

## 🏛️ End-to-End System & Cloud Architecture

```
┌─────────────────────────────────┐
│     SagarMitra Flutter App      │
│  (Android / iOS / Web / Edge)   │
│  - Live EC2 Health Polling      │
│  - Camera / Gallery Capture     │
│  - Interactive Zoom & Inspection│
│  - Historical Runs Gallery      │
└───────────────┬─────────────────┘
                │ HTTP REST / Multipart Upload
                ▼
┌────────────────────────────────────────────────────────┐
│             Amazon Web Services (AWS EC2)              │
│       Elastic IP: 13.200.225.184 (ap-south-1)          │
│                                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │   Dockerized FastAPI Microservice (Port 80)      │  │
│  │   - Uvicorn Asynchronous ASGI Server             │  │
│  │   - CUDA 11.8 / PyTorch 2.0.1 Runtime            │  │
│  │   - 4-Stage SagarMitra Inference Engine          │  │
│  │   - Auto-Restart Policy (`unless-stopped`)       │  │
│  └─────────────┬──────────────────────┬─────────────┘  │
│                │                      │                │
│                ▼                      ▼                │
│       Host-Mounted Volume        Amazon S3             │
│       `/data/detections.db`      Bucket: maritime-...  │
│       (SQLite Metadata Store)    (Annotated Artifacts) │
└────────────────────────────────────────────────────────┘
```

### Key Engineering Principles
- **Dual-Persistence Pattern:**
  - **Amazon S3 (`maritime-capstone-app-images-2026`):** Stores high-resolution rendered output images with bounding box overlays and confidence labels.
  - **Persistent Host SQLite (`/data/detections.db`):** Mounted outside the Docker container. Stores granular telemetry for every inference run (per-class counts, average confidence, wall-clock inference duration, hardware device, model checkpoint versions).
- **Research Export Readiness:** Built-in endpoints allow instant export of complete historical inference runs to CSV for empirical tables in academic papers.
- **Fail-Safe Containerization:** Configured with Docker's `unless-stopped` restart policy to ensure zero-touch server recovery after EC2 reboots or spot interruptions.

---

## 📱 Client Application (Flutter Material 3)

The user-facing mobile/edge client is built with **Flutter 3.x** and styled with **Material 3**:

- **Real-Time Connectivity Radar:** Dynamic status banner monitoring the live health of the EC2 cloud engine (`GET /`).
- **Interactive Inspection Studio:** Custom modal viewer supporting multi-touch pinch-to-zoom and pan for close inspection of distant objects in open water.
- **Dynamic Network Settings:** Configurable base URL that allows instant switching between the AWS Elastic IP and local edge development servers (`http://13.200.225.184` default).
- **Historical Gallery:** S3-synced visual history feed filtered strictly for valid image artifacts (`.jpg`, `.png`, `.webp`) with pull-to-refresh capabilities.

---

## 🔌 API Specification & Data Telemetry

### 1. Execute Inference Pipeline
```http
POST /detect?min_confidence=0.50
Content-Type: multipart/form-data
```
**Request:**
- `file`: Raw image binary (JPEG, PNG).
- `min_confidence`: Minimum confidence threshold (float $0.0 - 1.0$, default $0.50$).

**Response (`200 OK`):**
```json
{
  "success": true,
  "run_id": "76ed55e2-569c-4729-82ee-8458d41fe2f2",
  "filename": "76ed55e2-569c-4729-82ee-8458d41fe2f2.jpg",
  "image_url": "https://maritime-capstone-app-images-2026.s3.amazonaws.com/76ed55e2-569c-4729-82ee-8458d41fe2f2.jpg",
  "image_width": 1920,
  "image_height": 1080,
  "inference_time_s": 14.8210,
  "detections": [
    {
      "category_id": 1,
      "category": "swimmer",
      "score": 0.8842,
      "bbox": [842.1, 412.5, 24.0, 18.2]
    },
    {
      "category_id": 4,
      "category": "life saving appliance",
      "score": 0.9215,
      "bbox": [880.0, 420.1, 14.5, 15.0]
    }
  ],
  "message": "Image processed, uploaded, and metadata logged successfully."
}
```

### 2. Research Paper Telemetry & Stats
```http
GET /stats
```
**Response (`200 OK`):**
```json
{
  "summary": {
    "total_runs": 142,
    "total_objects": 864,
    "total_swimmers": 312,
    "total_boats": 420,
    "total_jetskis": 45,
    "total_lsas": 55,
    "total_buoys": 32,
    "overall_avg_confidence": 0.8142,
    "avg_inference_time_s": 15.21
  },
  "per_class": [
    {"category": "swimmer", "count": 312, "avg_score": 0.792},
    {"category": "boat", "count": 420, "avg_score": 0.865}
  ]
}
```

### 3. Additional Endpoints
- `GET /records?limit=50&offset=0`: Paginated runs metadata from SQLite.
- `GET /records/{run_id}/detections`: Granular bounding-box telemetry for a specific run.
- `GET /export/csv`: Direct streaming download of all historical inference data for LaTeX/Excel ingestion.
- `GET /history`: S3 image feed for mobile gallery preview.

---

## 📂 Repository Structure

```
CAPSTONE_APP/
├── App/                                # Cross-platform Flutter mobile client
│   ├── android/                        # Android platform build & permissions
│   ├── ios/                            # iOS platform configuration
│   ├── lib/
│   │   ├── config/                     # Palette, categories, shared preferences
│   │   ├── models/                     # Strongly-typed JSON deserializers
│   │   ├── screens/                    # Detect, History, and Settings views
│   │   ├── services/                   # HTTP client & AWS communication
│   │   └── widgets/                    # Status radar, preview modal, chips
│   └── pubspec.yaml                    # Flutter dependencies
│
├── backend/                            # Cloud AI inference engine
│   ├── models/                         # Checkpoints (CBNetV2, YOLO11s, Refiners)
│   ├── runtime/                        # Lightweight inference execution runtime
│   ├── source/                         # Custom neural network layer definitions
│   ├── database.py                     # SQLite persistence engine
│   ├── pipeline_engine.py              # 4-stage staged detector implementation
│   ├── main.py                         # FastAPI REST application
│   ├── pipeline_config.json            # Exact model parameters & thresholds
│   ├── Dockerfile                      # CUDA 11.8 + PyTorch + MMCV container
│   ├── deploy_to_ec2.ps1               # Automated S3-to-SSM deployment automation
│   └── requirements.txt                # Python dependencies
│
├── start_server.cmd                    # One-click AWS EC2 startup script
├── stop_server.cmd                     # One-click AWS EC2 safe shutdown script
└── README.md                           # Master project documentation
```

---

## 🚀 Setup, Deployment & Reproduction Guide

### Prerequisites
- **Operating System:** Ubuntu 22.04 LTS (recommended for server) or Windows 10/11.
- **Hardware:** NVIDIA GPU with $\ge 8\text{ GB}$ VRAM (CUDA 11.8 compatible) for optimal inference throughput.
- **Tooling:** Docker Engine, AWS CLI (configured), Flutter SDK $\ge 3.19$.

### Option A: Local Python Inference Environment
```bash
# 1. Clone repository
git clone https://github.com/gourav05052004/CAPSTONE_APP.git
cd CAPSTONE_APP/backend

# 2. Create isolated Python 3.11 environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# 3. Install strict PyTorch and MMCV dependencies
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install openmim
mim install mmcv-full==1.7.2 mmdet==2.28.2
pip install -r requirements.txt

# 4. Verify bundle integrity
python verify_bundle.py

# 5. Start API server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Option B: Production Docker Container
```bash
cd CAPSTONE_APP/backend

# Build the complete CUDA-enabled image
docker build -t sagarmitra-backend .

# Run with persistent volume for SQLite research records
docker run -d --name sagarmitra-api \
  --restart unless-stopped \
  -p 80:8000 \
  -v /data:/data \
  -e AWS_ACCESS_KEY_ID="<YOUR_KEY>" \
  -e AWS_SECRET_ACCESS_KEY="<YOUR_SECRET>" \
  -e AWS_DEFAULT_REGION="ap-south-1" \
  sagarmitra-backend
```

### Option C: Running the Mobile App (Flutter)
```bash
cd CAPSTONE_APP/App

# Fetch dependencies
flutter pub get

# Launch on connected mobile device or desktop
flutter run
```

---





