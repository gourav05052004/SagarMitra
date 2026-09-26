import 'package:flutter/material.dart';

class ImagePreviewDialog extends StatelessWidget {
  final String imageUrl;
  final String title;

  const ImagePreviewDialog({
    super.key,
    required this.imageUrl,
    this.title = "Detection Result",
  });

  static void show(BuildContext context, String imageUrl, {String? title}) {
    showDialog(
      context: context,
      builder: (_) => ImagePreviewDialog(
        imageUrl: imageUrl,
        title: title ?? "Detection Result",
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Dialog(
      backgroundColor: Colors.black,
      insetPadding: const EdgeInsets.all(8),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(16),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Header Bar
            Container(
              color: Colors.black87,
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Expanded(
                    child: Text(
                      title,
                      style: const TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                        fontSize: 14,
                      ),
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  IconButton(
                    icon: const Icon(Icons.close, color: Colors.white),
                    onPressed: () => Navigator.of(context).pop(),
                  ),
                ],
              ),
            ),

            // Zoomable Interactive Image
            Flexible(
              child: Container(
                color: Colors.black,
                constraints: BoxConstraints(
                  maxHeight: MediaQuery.of(context).size.height * 0.75,
                ),
                child: InteractiveViewer(
                  panEnabled: true,
                  boundaryMargin: const EdgeInsets.all(20),
                  minScale: 0.5,
                  maxScale: 4.0,
                  child: Center(
                    child: Image.network(
                      imageUrl,
                      fit: BoxFit.contain,
                      loadingBuilder: (context, child, loadingProgress) {
                        if (loadingProgress == null) return child;
                        final total = loadingProgress.expectedTotalBytes;
                        final loaded = loadingProgress.cumulativeBytesLoaded;
                        return Center(
                          child: CircularProgressIndicator(
                            value: total != null ? loaded / total : null,
                            color: Colors.cyanAccent,
                          ),
                        );
                      },
                      errorBuilder: (context, error, stackTrace) {
                        return const Center(
                          child: Padding(
                            padding: EdgeInsets.all(24.0),
                            child: Column(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Icon(Icons.broken_image, color: Colors.redAccent, size: 48),
                                SizedBox(height: 8),
                                Text(
                                  "Failed to load image from S3",
                                  style: TextStyle(color: Colors.white70),
                                ),
                              ],
                            ),
                          ),
                        );
                      },
                    ),
                  ),
                ),
              ),
            ),

            // Bottom tip
            Container(
              color: Colors.black87,
              padding: const EdgeInsets.symmetric(vertical: 8),
              child: const Center(
                child: Text(
                  "Pinch to zoom • Drag to pan",
                  style: TextStyle(color: Colors.white54, fontSize: 12),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
