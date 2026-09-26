import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import '../config/app_config.dart';
import '../models/detection_response.dart';
import '../models/history_item.dart';

class ApiException implements Exception {
  final String message;
  final int? statusCode;

  ApiException(this.message, [this.statusCode]);

  @override
  String toString() => message;
}

class ApiService {
  final http.Client _client;

  ApiService({http.Client? client}) : _client = client ?? http.Client();

  /// Check backend connectivity and status
  Future<bool> checkHealth() async {
    try {
      final baseUrl = await AppConfig.getBaseUrl();
      final url = Uri.parse('$baseUrl/');
      final response = await _client.get(url).timeout(const Duration(seconds: 7));
      
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return data['message'] != null;
      }
      return false;
    } catch (_) {
      return false;
    }
  }

  /// Upload an image file for maritime detection inference
  Future<DetectionResponse> detectObjects(File imageFile) async {
    final baseUrl = await AppConfig.getBaseUrl();
    final url = Uri.parse('$baseUrl/detect');

    if (!await imageFile.exists()) {
      throw ApiException("Selected image file does not exist.");
    }

    try {
      final request = http.MultipartRequest('POST', url);
      
      request.files.add(
        await http.MultipartFile.fromPath('file', imageFile.path),
      );

      // 60-second timeout to allow multi-model inference pipeline
      final streamedResponse = await _client.send(request).timeout(
        const Duration(seconds: 60),
        onTimeout: () {
          throw ApiException(
            "Inference timed out. The server might still be warming up the model or the instance is under load.",
          );
        },
      );

      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final decoded = jsonDecode(response.body) as Map<String, dynamic>;
        return DetectionResponse.fromJson(decoded);
      } else {
        String serverError = "Server returned status ${response.statusCode}";
        try {
          final errorData = jsonDecode(response.body);
          if (errorData['error'] != null) {
            serverError = errorData['error'].toString();
          }
        } catch (_) {}
        throw ApiException(serverError, response.statusCode);
      }
    } on SocketException {
      throw ApiException(
        "Cannot reach the server at $baseUrl. Please verify the EC2 instance is running and the Elastic IP is active.",
      );
    } on FormatException {
      throw ApiException("Failed to parse response from server.");
    } catch (e) {
      if (e is ApiException) rethrow;
      throw ApiException("Failed to detect objects: $e");
    }
  }

  /// Fetch previous detections history
  Future<List<HistoryItem>> getHistory({int limit = 20}) async {
    final baseUrl = await AppConfig.getBaseUrl();
    final url = Uri.parse('$baseUrl/history?limit=$limit');

    try {
      final response = await _client.get(url).timeout(const Duration(seconds: 15));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        final rawList = data['history'] as List? ?? [];
        return rawList
            .map((item) => HistoryItem.fromJson(item as Map<String, dynamic>))
            .toList();
      } else {
        throw ApiException("Failed to load history (Status ${response.statusCode})");
      }
    } on SocketException {
      throw ApiException("Network error: Server unreachable.");
    } catch (e) {
      if (e is ApiException) rethrow;
      throw ApiException("Error loading history: $e");
    }
  }
}
