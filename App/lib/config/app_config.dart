import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class AppConfig {
  static const String defaultBaseUrl = "http://13.200.225.184";
  static const String _baseUrlPrefKey = "maritime_api_base_url";

  static final Map<int, String> categoryNames = {
    1: "Swimmer",
    2: "Boat",
    3: "Jetski",
    4: "Life Saving Appliance",
    5: "Buoy",
  };

  static final Map<int, Color> categoryColors = {
    1: Colors.orange,       // Swimmer
    2: Colors.blueAccent,   // Boat
    3: Colors.teal,         // Jetski
    4: Colors.redAccent,    // Life Saving Appliance
    5: Colors.amber,        // Buoy
  };

  static final Map<int, IconData> categoryIcons = {
    1: Icons.pool,
    2: Icons.directions_boat,
    3: Icons.speed,
    4: Icons.emergency,
    5: Icons.fmd_good,
  };

  static String getCategoryName(int categoryId) {
    return categoryNames[categoryId] ?? "Object ($categoryId)";
  }

  static Color getCategoryColor(int categoryId) {
    return categoryColors[categoryId] ?? Colors.blueGrey;
  }

  static IconData getCategoryIcon(int categoryId) {
    return categoryIcons[categoryId] ?? Icons.category;
  }

  static Future<String> getBaseUrl() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_baseUrlPrefKey) ?? defaultBaseUrl;
  }

  static Future<void> setBaseUrl(String newUrl) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_baseUrlPrefKey, newUrl.trim());
  }

  static Future<void> resetBaseUrl() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_baseUrlPrefKey);
  }
}
