import 'package:flutter/material.dart';
import '../config/app_config.dart';
import '../services/api_service.dart';

class SettingsScreen extends StatefulWidget {
  final ApiService apiService;
  final VoidCallback onConfigChanged;

  const SettingsScreen({
    super.key,
    required this.apiService,
    required this.onConfigChanged,
  });

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final TextEditingController _urlController = TextEditingController();
  bool _isTesting = false;
  String? _testResult;
  bool? _testSuccess;

  @override
  void initState() {
    super.initState();
    _loadCurrentUrl();
  }

  Future<void> _loadCurrentUrl() async {
    final currentUrl = await AppConfig.getBaseUrl();
    setState(() {
      _urlController.text = currentUrl;
    });
  }

  Future<void> _saveUrl() async {
    final newUrl = _urlController.text.trim();
    if (newUrl.isEmpty) return;

    await AppConfig.setBaseUrl(newUrl);
    widget.onConfigChanged();
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Base URL updated successfully")),
      );
    }
  }

  Future<void> _resetUrl() async {
    await AppConfig.resetBaseUrl();
    await _loadCurrentUrl();
    widget.onConfigChanged();
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Reset to default AWS Elastic IP")),
      );
    }
  }

  Future<void> _testConnection() async {
    setState(() {
      _isTesting = true;
      _testResult = null;
      _testSuccess = null;
    });

    final ok = await widget.apiService.checkHealth();

    if (mounted) {
      setState(() {
        _isTesting = false;
        _testSuccess = ok;
        _testResult = ok
            ? "Connection successful! Maritime Detection API is active."
            : "Cannot reach backend. Check if EC2 is running and security groups allow HTTP.";
      });
    }
  }

  @override
  void dispose() {
    _urlController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Server Configuration Card
          Card(
            elevation: 2,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Row(
                    children: [
                      Icon(Icons.dns, color: Color(0xFF0F4C81)),
                      SizedBox(width: 8),
                      Text(
                        "Backend Endpoint",
                        style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  const Text(
                    "AWS Elastic IP endpoint for maritime inference service.",
                    style: TextStyle(color: Colors.black54, fontSize: 13),
                  ),
                  const SizedBox(height: 16),
                  TextField(
                    controller: _urlController,
                    decoration: InputDecoration(
                      labelText: "Base URL",
                      hintText: "http://13.200.225.184",
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
                      suffixIcon: IconButton(
                        icon: const Icon(Icons.save),
                        onPressed: _saveUrl,
                        tooltip: "Save URL",
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: _resetUrl,
                          icon: const Icon(Icons.restart_alt),
                          label: const Text("Reset Default"),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: ElevatedButton.icon(
                          onPressed: _isTesting ? null : _testConnection,
                          icon: _isTesting
                              ? const SizedBox(
                                  width: 16,
                                  height: 16,
                                  child: CircularProgressIndicator(strokeWidth: 2),
                                )
                              : const Icon(Icons.network_check),
                          label: const Text("Test Ping"),
                        ),
                      ),
                    ],
                  ),
                  if (_testResult != null) ...[
                    const SizedBox(height: 12),
                    Container(
                      padding: const EdgeInsets.all(10),
                      decoration: BoxDecoration(
                        color: _testSuccess == true ? Colors.green.shade50 : Colors.red.shade50,
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Row(
                        children: [
                          Icon(
                            _testSuccess == true ? Icons.check_circle : Icons.warning,
                            color: _testSuccess == true ? Colors.green : Colors.red,
                            size: 18,
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              _testResult!,
                              style: TextStyle(
                                fontSize: 12,
                                color: _testSuccess == true ? Colors.green.shade900 : Colors.red.shade900,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ),

          const SizedBox(height: 16),

          // Maritime Classes Supported Card
          Card(
            elevation: 2,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    "Target Maritime Classes",
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 12),
                  ...AppConfig.categoryNames.entries.map((entry) {
                    final color = AppConfig.getCategoryColor(entry.key);
                    final icon = AppConfig.getCategoryIcon(entry.key);
                    return Padding(
                      padding: const EdgeInsets.symmetric(vertical: 4.0),
                      child: Row(
                        children: [
                          Icon(icon, color: color, size: 20),
                          const SizedBox(width: 12),
                          Text(
                            entry.value,
                            style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w500),
                          ),
                          const Spacer(),
                          Text(
                            "ID: ${entry.key}",
                            style: const TextStyle(fontSize: 12, color: Colors.black45),
                          ),
                        ],
                      ),
                    );
                  }),
                ],
              ),
            ),
          ),

          const SizedBox(height: 16),

          // Pipeline Architecture Info
          const Card(
            elevation: 2,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.all(Radius.circular(16))),
            child: Padding(
              padding: EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    "Pipeline Architecture",
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                  ),
                  SizedBox(height: 8),
                  Text("• Benchmark Score: 61.5735 COCO bbox AP50:95"),
                  Text("• Primary: CBNetV2–Swin-S Cascade R-CNN"),
                  Text("• Refinement: Class-safe & pixel-boundary correction"),
                  Text("• Expert Fusion: Regression-detail YOLO11s"),
                  Text("• Tiled Correction: Life-saving appliances & swimmers"),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
