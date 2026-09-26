import '../config/app_config.dart';

class DetectionItem {
  final int categoryId;
  final String label;
  final double score;
  final List<double> bbox; // [x, y, width, height]

  DetectionItem({
    required this.categoryId,
    required this.label,
    required this.score,
    required this.bbox,
  });

  factory DetectionItem.fromJson(Map<String, dynamic> json) {
    // Determine category ID & label
    int catId = 0;
    String name = "Unknown";

    if (json.containsKey('category_id') && json['category_id'] != null) {
      catId = int.tryParse(json['category_id'].toString()) ?? 0;
      name = AppConfig.getCategoryName(catId);
    } else if (json.containsKey('label') && json['label'] != null) {
      name = json['label'].toString();
      // Reverse map category name to id if possible
      for (final entry in AppConfig.categoryNames.entries) {
        if (entry.value.toLowerCase() == name.toLowerCase()) {
          catId = entry.key;
          break;
        }
      }
    }

    // Determine confidence/score
    double conf = 0.0;
    if (json.containsKey('confidence') && json['confidence'] != null) {
      conf = double.tryParse(json['confidence'].toString()) ?? 0.0;
    } else if (json.containsKey('score') && json['score'] != null) {
      conf = double.tryParse(json['score'].toString()) ?? 0.0;
    }

    // Determine bbox coordinates [x, y, w, h]
    List<double> box = [];
    if (json['bbox'] is List) {
      box = (json['bbox'] as List)
          .map((e) => double.tryParse(e.toString()) ?? 0.0)
          .toList();
    }

    return DetectionItem(
      categoryId: catId,
      label: name,
      score: conf,
      bbox: box,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'category_id': categoryId,
      'label': label,
      'score': score,
      'bbox': bbox,
    };
  }
}
