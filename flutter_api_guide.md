# Maritime Object Detection API - Flutter Integration Guide

This document outlines the API endpoints available on the Maritime Object Detection Backend and provides examples of how to consume them in a Flutter application.

## Base URL
Replace `13.200.225.184` with your AWS Elastic IP address. Since you have configured an Elastic IP, this address will remain static even if you stop and start your server to save credits!

```dart
const String baseUrl = "http://13.200.225.184";
```

---

## 1. Health Check
Check if the API is running and accessible.

- **URL:** `/`
- **Method:** `GET`
- **Response:**
  ```json
  {
    "message": "Maritime Object Detection API is running"
  }
  ```

### Flutter Example:
```dart
import 'package:http/http.dart' as http;
import 'dart:convert';

Future<bool> checkApiHealth() async {
  final response = await http.get(Uri.parse('$baseUrl/'));
  if (response.statusCode == 200) {
    print(jsonDecode(response.body)['message']);
    return true;
  }
  return false;
}
```

---

## 2. Detect Objects (Inference)
Upload an image to perform maritime object detection. The backend processes the image, draws bounding boxes, uploads the result to S3, and returns the public URL and detection details.

- **URL:** `/detect`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Body Parameter:** `file` (The image file)
- **Response:**
  ```json
  {
      "success": true,
      "image_url": "https://maritime-capstone-app-images-2026.s3.amazonaws.com/uuid.jpg",
      "filename": "uuid.jpg",
      "detections": [
          {
              "label": "boat",
              "confidence": 0.95,
              "bbox": [100.0, 150.0, 300.0, 400.0]
          }
      ],
      "message": "Image processed and uploaded successfully."
  }
  ```

### Flutter Example:
```dart
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:io';

Future<Map<String, dynamic>?> uploadImageForDetection(File imageFile) async {
  var request = http.MultipartRequest('POST', Uri.parse('$baseUrl/detect'));
  
  // Add the file to the request
  request.files.add(
    await http.MultipartFile.fromPath('file', imageFile.path)
  );

  try {
    var streamedResponse = await request.send();
    var response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode == 200) {
      var data = jsonDecode(response.body);
      print("Success! Annotated Image URL: ${data['image_url']}");
      return data;
    } else {
      print("Error: ${response.statusCode} - ${response.body}");
      return null;
    }
  } catch (e) {
    print("Failed to upload image: $e");
    return null;
  }
}
```

---

## 3. Get Inference History
Retrieve a list of all previously processed images and their public S3 URLs.

- **URL:** `/history`
- **Method:** `GET`
- **Query Parameter (Optional):** `limit` (default is 20)
- **Response:**
  ```json
  {
      "history": [
          {
              "filename": "uuid.jpg",
              "image_url": "https://maritime-capstone-app-images-2026.s3.amazonaws.com/uuid.jpg",
              "timestamp": "2026-09-26T16:05:00.000Z"
          }
      ]
  }
  ```

### Flutter Example:
```dart
import 'package:http/http.dart' as http;
import 'dart:convert';

Future<List<dynamic>> getHistory({int limit = 20}) async {
  final response = await http.get(Uri.parse('$baseUrl/history?limit=$limit'));
  
  if (response.statusCode == 200) {
    var data = jsonDecode(response.body);
    return data['history'] as List<dynamic>;
  } else {
    print("Failed to fetch history");
    return [];
  }
}
```

---

### Tips for your Flutter Developer:
1. **Permissions:** Ensure the Flutter app has file system and camera permissions (e.g., `image_picker` package) in `AndroidManifest.xml` and `Info.plist`.
2. **Network Package:** The `http` package (`flutter pub add http`) is required for the examples above.
3. **Static IP Address:** The backend is configured with an AWS Elastic IP. The `baseUrl` can be hardcoded since the IP will not change when the server is stopped and started.
