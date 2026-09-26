import 'detection_item.dart';

class DetectionResponse {
  final bool success;
  final String imageUrl;
  final String filename;
  final List<DetectionItem> detections;
  final String message;

  DetectionResponse({
    required this.success,
    required this.imageUrl,
    required this.filename,
    required this.detections,
    required this.message,
  });

  factory DetectionResponse.fromJson(Map<String, dynamic> json) {
    var rawList = json['detections'] as List? ?? [];
    List<DetectionItem> parsedDetections = rawList
        .map((item) => DetectionItem.fromJson(item as Map<String, dynamic>))
        .toList();

    return DetectionResponse(
      success: json['success'] ?? true,
      imageUrl: json['image_url'] ?? '',
      filename: json['filename'] ?? '',
      detections: parsedDetections,
      message: json['message'] ?? '',
    );
  }
}
