class HistoryItem {
  final String filename;
  final String imageUrl;
  final DateTime? timestamp;

  HistoryItem({
    required this.filename,
    required this.imageUrl,
    this.timestamp,
  });

  factory HistoryItem.fromJson(Map<String, dynamic> json) {
    DateTime? dt;
    if (json['timestamp'] != null) {
      dt = DateTime.tryParse(json['timestamp'].toString());
    }

    return HistoryItem(
      filename: json['filename'] ?? '',
      imageUrl: json['image_url'] ?? '',
      timestamp: dt,
    );
  }
}
