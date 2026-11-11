import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:photo_manager/photo_manager.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

void main() => runApp(const FotoRunApp());

class FotoRunApp extends StatelessWidget {
  const FotoRunApp({super.key});
  @override
  Widget build(BuildContext c) {
    ColorScheme s(Brightness b) =>
        ColorScheme.fromSeed(seedColor: const Color(0xFF6C5CE7), brightness: b);
    return MaterialApp(
      title: 'FotoRun',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(useMaterial3: true, colorScheme: s(Brightness.light)),
      darkTheme: ThemeData(useMaterial3: true, colorScheme: s(Brightness.dark)),
      home: const Home(),
    );
  }
}

// ───────────────────────── Modelo de datos ─────────────────────────

class Person {
  String name;
  List<List<double>> embeddings;
  Person(this.name,
