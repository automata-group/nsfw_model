import 'dart:io';
import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_image_compress/flutter_image_compress.dart';
import '../models/moderation_result.dart';

class ModerationService {
  final Dio _dio;
  final String baseUrl;

  ModerationService({this.baseUrl = 'http://10.0.2.2:8080'})
      : _dio = Dio(BaseOptions(
          baseUrl: baseUrl,
          connectTimeout: const Duration(seconds: 4),
          receiveTimeout: const Duration(seconds: 6),
          headers: {'Accept': 'application/json'},
        ));

  static Future<Uint8List> compressImageInIsolate(String filePath) async {
    final result = await FlutterImageCompress.compressWithFile(
      filePath,
      minWidth: 512,
      minHeight: 512,
      quality: 85,
      format: CompressFormat.jpeg,
    );
    if (result == null) {
      throw Exception('Failed compressing image in isolate');
    }
    return result;
  }

  Future<ModerationResult> moderateImage(String filePath) async {
    final compressedBytes = await compute(compressImageInIsolate, filePath);

    final formData = FormData.fromMap({
      'image': MultipartFile.fromBytes(
        compressedBytes,
        filename: 'upload.jpg',
      ),
    });

    try {
      final response = await _dio.post(
        '/api/v1/moderate',
        data: formData,
        options: Options(contentType: 'multipart/form-data'),
      );

      if (response.statusCode == 200) {
        return ModerationResult.fromJson(response.data as Map<String, dynamic>);
      } else {
        throw Exception('Server returned HTTP ${response.statusCode}');
      }
    } on DioException catch (e) {
      final msg = e.response?.data?['error'] ?? e.message;
      throw Exception('Network error during moderation: $msg');
    }
  }
}
