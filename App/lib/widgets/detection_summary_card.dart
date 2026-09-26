import 'package:flutter/material.dart';
import '../config/app_config.dart';
import '../models/detection_item.dart';

class DetectionSummaryCard extends StatelessWidget {
  final List<DetectionItem> detections;

  const DetectionSummaryCard({super.key, required this.detections});

  @override
  Widget build(BuildContext context) {
    if (detections.isEmpty) {
      return Card(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        color: Colors.grey.shade50,
        child: const Padding(
          padding: EdgeInsets.all(16.0),
          child: Row(
            children: [
              Icon(Icons.info_outline, color: Colors.grey),
              SizedBox(width: 12),
              Text(
                "No maritime objects detected in this image.",
                style: TextStyle(color: Colors.black87),
              ),
            ],
          ),
        ),
      );
    }

    // Group detections by category
    final Map<int, int> counts = {};
    for (var d in detections) {
      counts[d.categoryId] = (counts[d.categoryId] ?? 0) + 1;
    }

    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  "Detection Breakdown",
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: Colors.blue.shade100,
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text(
                    "${detections.length} Found",
                    style: TextStyle(
                      color: Colors.blue.shade900,
                      fontWeight: FontWeight.bold,
                      fontSize: 12,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),

            // Chips summarizing each detected class count
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: counts.entries.map((entry) {
                final color = AppConfig.getCategoryColor(entry.key);
                final name = AppConfig.getCategoryName(entry.key);
                final icon = AppConfig.getCategoryIcon(entry.key);

                return Chip(
                  avatar: Icon(icon, size: 16, color: color),
                  label: Text(
                    "$name: ${entry.value}",
                    style: TextStyle(
                      color: Colors.black87,
                      fontWeight: FontWeight.w600,
                      fontSize: 12,
                    ),
                  ),
                  backgroundColor: color.withOpacity(0.15),
                  side: BorderSide(color: color.withOpacity(0.5)),
                  padding: const EdgeInsets.symmetric(horizontal: 4),
                );
              }).toList(),
            ),

            const Divider(height: 24),

            const Text(
              "Detected Instances",
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w600,
                color: Colors.black54,
              ),
            ),
            const SizedBox(height: 8),

            // Detailed instance list
            ListView.separated(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              itemCount: detections.length,
              separatorBuilder: (_, __) => const Divider(height: 1),
              itemBuilder: (context, index) {
                final item = detections[index];
                final color = AppConfig.getCategoryColor(item.categoryId);
                final icon = AppConfig.getCategoryIcon(item.categoryId);
                final pct = (item.score * 100).toStringAsFixed(1);

                return ListTile(
                  contentPadding: EdgeInsets.zero,
                  leading: CircleAvatar(
                    backgroundColor: color.withOpacity(0.2),
                    child: Icon(icon, color: color, size: 20),
                  ),
                  title: Text(
                    item.label,
                    style: const TextStyle(fontWeight: FontWeight.w600),
                  ),
                  subtitle: item.bbox.length == 4
                      ? Text(
                          "Box: [${item.bbox[0].toInt()}, ${item.bbox[1].toInt()}, ${item.bbox[2].toInt()}x${item.bbox[3].toInt()}]",
                          style: const TextStyle(fontSize: 11),
                        )
                      : null,
                  trailing: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: item.score >= 0.7
                          ? Colors.green.shade50
                          : (item.score >= 0.4 ? Colors.orange.shade50 : Colors.red.shade50),
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(
                        color: item.score >= 0.7
                            ? Colors.green.shade300
                            : (item.score >= 0.4 ? Colors.orange.shade300 : Colors.red.shade300),
                      ),
                    ),
                    child: Text(
                      "$pct%",
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                        color: item.score >= 0.7
                            ? Colors.green.shade800
                            : (item.score >= 0.4 ? Colors.orange.shade800 : Colors.red.shade800),
                      ),
                    ),
                  ),
                );
              },
            ),
          ],
        ),
      ),
    );
  }
}
