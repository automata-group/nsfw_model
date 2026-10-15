import 'package:flutter/material.dart';
import 'screens/moderation_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const NsfwGuardApp());
}

class NsfwGuardApp extends StatelessWidget {
  const NsfwGuardApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'NSFW AI Content Moderator',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF090D16),
        primaryColor: const Color(0xFF6366F1),
      ),
      home: const ModerationScreen(),
    );
  }
}
