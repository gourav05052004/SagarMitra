import 'package:flutter/material.dart';

class StatusBanner extends StatelessWidget {
  final bool? isConnected;
  final bool isChecking;
  final VoidCallback onRefresh;

  const StatusBanner({
    super.key,
    required this.isConnected,
    required this.isChecking,
    required this.onRefresh,
  });

  @override
  Widget build(BuildContext context) {
    Color bgColor;
    Color textColor;
    IconData icon;
    String text;

    if (isChecking) {
      bgColor = Colors.blue.shade50;
      textColor = Colors.blue.shade900;
      icon = Icons.sync;
      text = "Checking server status...";
    } else if (isConnected == true) {
      bgColor = Colors.green.shade50;
      textColor = Colors.green.shade900;
      icon = Icons.cloud_done;
      text = "Backend Online (AWS EC2 Active)";
    } else {
      bgColor = Colors.amber.shade50;
      textColor = Colors.amber.shade900;
      icon = Icons.cloud_off;
      text = "Backend Offline / Unreachable";
    }

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      color: bgColor,
      child: Row(
        children: [
          Icon(icon, size: 18, color: textColor),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              text,
              style: TextStyle(
                color: textColor,
                fontSize: 13,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
          IconButton(
            icon: Icon(Icons.refresh, size: 18, color: textColor),
            onPressed: isChecking ? null : onRefresh,
            tooltip: "Check connectivity",
            constraints: const BoxConstraints(),
            padding: const EdgeInsets.all(4),
          ),
        ],
      ),
    );
  }
}
