import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import '../models/moderation_result.dart';
import '../services/moderation_service.dart';

class ModerationScreen extends StatefulWidget {
  const ModerationScreen({super.key});

  @override
  State<ModerationScreen> createState() => _ModerationScreenState();
}

class _ModerationScreenState extends State<ModerationScreen> {
  final ModerationService _service = ModerationService();
  final ImagePicker _picker = ImagePicker();

  File? _selectedImage;
  ModerationResult? _result;
  bool _isLoading = false;
  String? _errorMessage;

  Future<void> _pickAndModerate(ImageSource source) async {
    try {
      final picked = await _picker.pickImage(source: source);
      if (picked == null) return;

      setState(() {
        _selectedImage = File(picked.path);
        _isLoading = true;
        _errorMessage = null;
        _result = null;
      });

      final res = await _service.moderateImage(picked.path);

      setState(() {
        _result = res;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _errorMessage = e.toString();
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF090D16),
      appBar: AppBar(
        title: Text(
          'فاحص المحتوى الذكي | AI Guard',
          style: GoogleFonts.cairo(fontWeight: FontWeight.bold, fontSize: 18),
        ),
        backgroundColor: const Color(0xFF111827),
        elevation: 0,
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            _buildImageCard(),
            const SizedBox(height: 20),
            _buildActionButtons(),
            const SizedBox(height: 24),
            if (_isLoading)
              const Center(
                child: CircularProgressIndicator(color: Color(0xFF6366F1)),
              )
            else if (_errorMessage != null)
              _buildErrorBanner()
            else if (_result != null)
              _buildResultCard(_result!),
          ],
        ),
      ),
    );
  }

  Widget _buildImageCard() {
    return Container(
      width: double.infinity,
      height: 240,
      decoration: BoxDecoration(
        color: const Color(0xFF161F30),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withOpacity(0.08)),
      ),
      child: _selectedImage != null
          ? ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: Image.file(_selectedImage!, fit: BoxFit.cover),
            )
          : Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Icon(Icons.add_photo_alternate_outlined,
                    size: 54, color: Colors.white38),
                const SizedBox(height: 12),
                Text(
                  'قم باختيار صورة لفحصها فورياً',
                  style: GoogleFonts.cairo(color: Colors.white54, fontSize: 14),
                ),
              ],
            ),
    );
  }

  Widget _buildActionButtons() {
    return Row(
      children: [
        Expanded(
          child: ElevatedButton.icon(
            onPressed: () => _pickAndModerate(ImageSource.gallery),
            icon: const Icon(Icons.photo_library),
            label: Text('المعرض', style: GoogleFonts.cairo(fontWeight: FontWeight.w600)),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF4F46E5),
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(vertical: 14),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            ),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: ElevatedButton.icon(
            onPressed: () => _pickAndModerate(ImageSource.camera),
            icon: const Icon(Icons.camera_alt),
            label: Text('الكاميرا', style: GoogleFonts.cairo(fontWeight: FontWeight.w600)),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF1F2937),
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(vertical: 14),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildErrorBanner() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF7F1D1D).withOpacity(0.4),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFFEF4444)),
      ),
      child: Row(
        children: [
          const Icon(Icons.error_outline, color: Color(0xFFEF4444)),
          const SizedBox(width: 12),
          Expanded(
            child: Text(_errorMessage!, style: GoogleFonts.cairo(color: Colors.white)),
          ),
        ],
      ),
    );
  }

  Widget _buildResultCard(ModerationResult res) {
    final isSafe = !res.isNsfw;
    final color = isSafe ? const Color(0xFF10B981) : const Color(0xFFEF4444);

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF131B2E),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color.withOpacity(0.4)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                children: [
                  Icon(isSafe ? Icons.check_circle : Icons.warning_rounded, color: color, size: 28),
                  const SizedBox(width: 10),
                  Text(
                    isSafe ? 'محتوى آمن (Safe)' : 'محتوى غير لائق (NSFW)',
                    style: GoogleFonts.cairo(fontSize: 17, fontWeight: FontWeight.bold, color: color),
                  ),
                ],
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.06),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  '${res.latencyMs.toStringAsFixed(1)} ms',
                  style: const TextStyle(color: Colors.white70, fontSize: 12),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            'الفئة المحددة: ${res.category.toUpperCase()} (${(res.confidence * 100).toStringAsFixed(1)}%)',
            style: GoogleFonts.cairo(color: Colors.white, fontSize: 14, fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 4),
          Text(
            'التعليل الذكي: ${res.agentReasoning}',
            style: GoogleFonts.cairo(color: Colors.white60, fontSize: 13),
          ),
          const Divider(height: 30, color: Color(0xFF334155)),
          ...res.details.entries.map((entry) => _buildDetailBar(entry.key, entry.value)).toList(),
        ],
      ),
    ).animate().fadeIn(duration: 400.ms).slideY(begin: 0.1, end: 0);
  }

  Widget _buildDetailBar(String label, double prob) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          SizedBox(
            width: 80,
            child: Text(label, style: GoogleFonts.cairo(color: Colors.white70, fontSize: 13)),
          ),
          Expanded(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(6),
              child: LinearProgressIndicator(
                value: prob,
                backgroundColor: const Color(0xFF0F172A),
                color: const Color(0xFF6366F1),
                minHeight: 8,
              ),
            ),
          ),
          const SizedBox(width: 12),
          Text('${(prob * 100).toStringAsFixed(1)}%',
              style: GoogleFonts.cairo(color: Colors.white70, fontSize: 12)),
        ],
      ),
    );
  }
}
