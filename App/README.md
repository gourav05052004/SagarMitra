# Maritime Vision — Flutter Mobile Application

A cross-platform Flutter application for real-time maritime object detection, built for the Capstone project. It connects to the AWS EC2 hosted multi-stage Maritime Detection backend (CBNetV2 Swin-S + YOLO11s fusion).

---

## Features

- **Real-Time Backend Status:** Live health check banner connecting to `GET /` to verify if the AWS EC2 instance is reachable.
- **Image Detection (`POST /detect`):**
  - Capture photos with device camera or pick from image gallery.
  - Multi-stage model inference execution timer.
  - Displays high-resolution annotated image uploaded directly to Amazon S3.
  - Detailed detection breakdown chips and confidence percentages for:
    - *Swimmer*
    - *Boat*
    - *Jetski*
    - *Life Saving Appliance*
    - *Buoy*
- **Interactive Image Viewer:** Pinch-to-zoom and drag-to-pan modal viewer for inspection.
- **Inference History (`GET /history`):**
  - Scrollable thumbnail grid of previously processed images stored in S3.
  - Pull-to-refresh and formatted timestamps.
- **Settings & Dynamic IP Configuration:**
  - Configurable Base URL (defaults to AWS Elastic IP: `http://13.200.225.184`).
  - Quick connectivity ping testing tool.
  - Reset to default at any time.

---

## Directory Structure

```
App/
├── flutter_api_guide.md            # Backend API specification & Dart examples
├── README.md                       # App setup & documentation
├── pubspec.yaml                    # Flutter project configuration & dependencies
├── analysis_options.yaml           # Linting rules
├── android/
│   └── app/src/main/
│       └── AndroidManifest.xml     # Internet, Cleartext HTTP & Camera permissions
├── ios/
│   └── Runner/
│       └── Info.plist              # Camera, Photos & App Transport Security
└── lib/
    ├── main.dart                   # Application entry point & Material 3 theme
    ├── config/
    │   └── app_config.dart         # Base URL, Category IDs, Colors, SharedPreferences
    ├── models/
    │   ├── detection_item.dart     # Resilient parser for detection boxes & labels
    │   ├── detection_response.dart # Full /detect payload
    │   └── history_item.dart       # S3 history item parser
    ├── services/
    │   └── api_service.dart        # Multipart HTTP client for backend endpoints
    ├── screens/
    │   ├── main_navigation_screen.dart # Navigation bar & top status banner
    │   ├── detect_screen.dart          # Detection workflow screen
    │   ├── history_screen.dart         # S3 past inferences gallery
    │   └── settings_screen.dart        # IP config & connection tester
    └── widgets/
        ├── status_banner.dart          # Backend live/offline banner
        ├── detection_summary_card.dart # Classification badges & score metrics
        └── image_preview_dialog.dart   # Interactive zoom/pan modal
```

---

## Running the App

### 1. Prerequisites
- [Flutter SDK](https://flutter.dev/docs/get-started/install) installed and added to `PATH`.
- Android Studio / Xcode / Physical device or emulator.

### 2. Install Dependencies
```bash
cd App
flutter pub get
```

### 3. Launch App
```bash
flutter run
```
