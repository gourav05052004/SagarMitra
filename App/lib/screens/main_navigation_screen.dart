import 'package:flutter/material.dart';
import '../services/api_service.dart';
import '../widgets/status_banner.dart';
import 'detect_screen.dart';
import 'history_screen.dart';
import 'settings_screen.dart';

class MainNavigationScreen extends StatefulWidget {
  const MainNavigationScreen({super.key});

  @override
  State<MainNavigationScreen> createState() => _MainNavigationScreenState();
}

class _MainNavigationScreenState extends State<MainNavigationScreen> {
  int _currentIndex = 0;
  final ApiService _apiService = ApiService();

  bool? _isConnected;
  bool _isCheckingHealth = false;

  @override
  void initState() {
    super.initState();
    _checkServerHealth();
  }

  Future<void> _checkServerHealth() async {
    if (_isCheckingHealth) return;
    setState(() {
      _isCheckingHealth = true;
    });

    final ok = await _apiService.checkHealth();

    if (mounted) {
      setState(() {
        _isConnected = ok;
        _isCheckingHealth = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final List<Widget> screens = [
      DetectScreen(apiService: _apiService),
      HistoryScreen(apiService: _apiService),
      SettingsScreen(
        apiService: _apiService,
        onConfigChanged: _checkServerHealth,
      ),
    ];

    return Scaffold(
      appBar: AppBar(
        title: const Row(
          children: [
            Icon(Icons.sailing, color: Colors.white),
            SizedBox(width: 10),
            Text(
              "Maritime Object Detection",
              style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18),
            ),
          ],
        ),
        backgroundColor: const Color(0xFF0F4C81),
        foregroundColor: Colors.white,
        elevation: 0,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _checkServerHealth,
            tooltip: "Check Server Status",
          ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            StatusBanner(
              isConnected: _isConnected,
              isChecking: _isCheckingHealth,
              onRefresh: _checkServerHealth,
            ),
            Expanded(
              child: IndexedStack(
                index: _currentIndex,
                children: screens,
              ),
            ),
          ],
        ),
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _currentIndex,
        onDestinationSelected: (idx) {
          setState(() {
            _currentIndex = idx;
          });
        },
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.radar_outlined),
            selectedIcon: Icon(Icons.radar),
            label: "Detect",
          ),
          NavigationDestination(
            icon: Icon(Icons.history_outlined),
            selectedIcon: Icon(Icons.history),
            label: "History",
          ),
          NavigationDestination(
            icon: Icon(Icons.settings_outlined),
            selectedIcon: Icon(Icons.settings),
            label: "Settings",
          ),
        ],
      ),
    );
  }
}
