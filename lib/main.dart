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
import 'package:tflite_core/tflite_core.dart';

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
  Person(this.name, this.embeddings);
  Map<String, dynamic> toJson() => {'n': name, 'e': embeddings};
  factory Person.fromJson(Map<String, dynamic> j) => Person(
      j['n'],
      (j['e'] as List).map((v) => (v as List).map((x) => (x as num).toDouble()).toList()).toList());
}

class PeopleStore {
  static Future<List<Person>> load() async {
    final p = await SharedPreferences.getInstance();
    final raw = p.getString('people');
    if (raw == null) return [];
    return (jsonDecode(raw) as List).map((e) => Person.fromJson(e)).toList();
  }

  static Future<void> save(List<Person> list) async {
    final p = await SharedPreferences.getInstance();
    await p.setString('people', jsonEncode(list.map((e) => e.toJson()).toList()));
  }
}

// ───────────────────────── Reconocimiento facial ─────────────────────────

class FaceService {
  final _detector = FaceDetector(options: FaceDetectorOptions(performanceMode: FaceDetectorMode.accurate));
  Interpreter? _model;

  Future<void> init() async => _model ??= await Interpreter.fromAsset('assets/mobilefacenet.tflite');

  /// Devuelve el vector del rostro más grande de la imagen, o null si no hay rostro.
  Future<List<double>?> embed(File file) async {
    await init();
    final faces = await _detector.processImage(InputImage.fromFile(file));
    if (faces.isEmpty) return null;
    faces.sort((a, b) => (b.boundingBox.width * b.boundingBox.height)
        .compareTo(a.boundingBox.width * a.boundingBox.height));
    final box = faces.first.boundingBox;
    final src = img.decodeImage(await file.readAsBytes());
    if (src == null) return null;
    final x = max(0, box.left.toInt()), y = max(0, box.top.toInt());
    final w = min(box.width.toInt(), src.width - x), h = min(box.height.toInt(), src.height - y);
    final face = img.copyResize(img.copyCrop(src, x: x, y: y, width: w, height: h), width: 112, height: 112);

    final input = List.generate(1, (_) => List.generate(112, (yy) => List.generate(112, (xx) {
          final p = face.getPixel(xx, yy);
          return [(p.r - 127.5) / 128, (p.g - 127.5) / 128, (p.b - 127.5) / 128];
        })));
    final output = List.generate(1, (_) => List.filled(192, 0.0));
    _model!.run(input, output);
    final v = output[0];
    final norm = sqrt(v.fold<double>(0, (a, b) => a + b * b));
    return v.map((e) => e / norm).toList();
  }

  static double similarity(List<double> a, List<double> b) {
    double s = 0;
    for (var i = 0; i < a.length; i++) s += a[i] * b[i];
    return s; // vectores normalizados: coseno
  }
}

// ───────────────────────── Galería ─────────────────────────

class GalleryService {
  Future<bool> permission() async => (await PhotoManager.requestPermissionExtend()).hasAccess;

  Future<List<AssetEntity>> allAssets() async {
    final paths = await PhotoManager.getAssetPathList(type: RequestType.common, onlyAll: true);
    if (paths.isEmpty) return [];
    return paths.first.getAssetListRange(start: 0, end: await paths.first.assetCountAsync);
  }

  Future<Map<String, int>> albums() async {
    final paths = await PhotoManager.getAssetPathList(type: RequestType.common, onlyAll: false);
    return {for (final p in paths) p.name: await p.assetCountAsync};
  }

  /// Copias exactas: se agrupa por tamaño/fecha y se confirma con hash MD5.
  Future<List<List<AssetEntity>>> duplicates(List<AssetEntity> all) async {
    final rough = <String, List<AssetEntity>>{};
    for (final a in all) {
      rough.putIfAbsent('${a.width}x${a.height}-${a.createDateSecond}-${a.videoDuration.inSeconds}', () => []).add(a);
    }
    final result = <List<AssetEntity>>[];
    for (final g in rough.values.where((g) => g.length > 1)) {
      final byHash = <String, List<AssetEntity>>{};
      for (final a in g) {
        final f = await a.originFile;
        if (f == null) continue;
        byHash.putIfAbsent(md5.convert(await f.readAsBytes()).toString(), () => []).add(a);
      }
      result.addAll(byHash.values.where((x) => x.length > 1));
    }
    return result;
  }

  Future<void> delete(List<AssetEntity> list) => PhotoManager.editor.deleteWithIds(list.map((e) => e.id).toList());

  /// Solo Android. Mueve la foto a otro álbum existente.
  Future<void> moveTo(AssetEntity a, AssetPathEntity target) =>
      PhotoManager.editor.android.moveAssetToAnother(entity: a, target: target);

  /// Archivo analizable: la foto, o un fotograma si es video.
  Future<File?> analyzable(AssetEntity a) async {
    if (a.type == AssetType.image) return a.originFile;
    final bytes = await a.thumbnailDataWithSize(const ThumbnailSize(720, 720));
    if (bytes == null) return null;
    final f = File('${(await getTemporaryDirectory()).path}/${a.id.replaceAll('/', '_')}.jpg');
    return f.writeAsBytes(bytes);
  }
}

// ───────────────────────── Interfaz ─────────────────────────

class Home extends StatefulWidget {
  const Home({super.key});
  @override
  State<Home> createState() => _HomeState();
}

class _HomeState extends State<Home> {
  final gallery = GalleryService();
  final faces = FaceService();
  int tab = 0;
  List<AssetEntity> assets = [];
  Map<String, int> albums = {};
  List<List<AssetEntity>> dups = [];
  List<Person> people = [];
  double progress = 0;
  String status = 'Toca escanear para empezar';

  @override
  void initState() {
    super.initState();
    PeopleStore.load().then((p) => setState(() => people = p));
  }

  Future<void> scan() async {
    if (!await gallery.permission()) {
      setState(() => status = 'Sin permiso de galería');
      return;
    }
    setState(() => status = 'Escaneando…');
    assets = await gallery.allAssets();
    albums = await gallery.albums();
    dups = await gallery.duplicates(assets);
    setState(() => status = 'Listo: ${assets.length} archivos, ${dups.length} grupos de copias');
  }

  Future<void> searchPerson() async {
    if (assets.isEmpty) await scan();
    final pick = await ImagePicker().pickImage(source: ImageSource.gallery);
    if (pick == null) return;
    final ref = await faces.embed(File(pick.path));
    if (ref == null) {
      _msg('No encontré un rostro en esa foto');
      return;
    }
    final matches = <AssetEntity>[];
    for (var i = 0; i < assets.length; i++) {
      final f = await gallery.analyzable(assets[i]);
      final e = f == null ? null : await faces.embed(f);
      if (e != null && FaceService.similarity(ref, e) > 0.6) matches.add(assets[i]);
      if (i % 10 == 0) setState(() => progress = i / assets.length);
    }
    setState(() => progress = 0);
    if (!mounted) return;
    final confirmed = await Navigator.push<List<AssetEntity>>(
        context, MaterialPageRoute(builder: (_) => ConfirmPage(matches)));
    if (confirmed == null || confirmed.isEmpty) return;
    final name = await _askName();
    if (name == null) return;
    final embs = <List<double>>[ref];
    for (final a in confirmed) {
      final f = await gallery.analyzable(a);
      final e = f == null ? null : await faces.embed(f);
      if (e != null) embs.add(e);
    }
    people.add(Person(name, embs));
    await PeopleStore.save(people);
    setState(() {});
  }

  Future<String?> _askName() {
    final c = TextEditingController();
    return showDialog<String>(
        context: context,
        builder: (_) => AlertDialog(
              title: const Text('¿Cómo se llama?'),
              content: TextField(controller: c, autofocus: true),
              actions: [TextButton(onPressed: () => Navigator.pop(context, c.text.trim()), child: const Text('Guardar'))],
            ));
  }

  void _msg(String t) => ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(t)));

  @override
  Widget build(BuildContext context) {
    final pages = [_scanTab(), _peopleTab(), _analysisTab()];
    return Scaffold(
      appBar: AppBar(title: const Row(children: [Icon(Icons.photo_camera), Icon(Icons.search, size: 18), SizedBox(width: 8), Text('FotoRun', style: TextStyle(fontWeight: FontWeight.w800))])),
      body: Padding(padding: const EdgeInsets.all(16), child: pages[tab]),
      bottomNavigationBar: NavigationBar(
        selectedIndex: tab,
        onDestinationSelected: (i) => setState(() => tab = i),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.radar), label: 'Escanear'),
          NavigationDestination(icon: Icon(Icons.face), label: 'Personas'),
          NavigationDestination(icon: Icon(Icons.analytics_outlined), label: 'Análisis'),
        ],
      ),
    );
  }

  Widget _scanTab() => Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        FilledButton.icon(onPressed: scan, icon: const Icon(Icons.bolt), label: const Text('Escaneo rápido')),
        const SizedBox(height: 12),
        if (progress > 0) LinearProgressIndicator(value: progress),
        Text(status),
      ]);

  Widget _peopleTab() => Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        FilledButton.icon(onPressed: searchPerson, icon: const Icon(Icons.person_search), label: const Text('Buscar a una persona')),
        if (progress > 0) Padding(padding: const EdgeInsets.only(top: 12), child: LinearProgressIndicator(value: progress)),
        const SizedBox(height: 16),
        Wrap(spacing: 8, children: [for (final p in people) Chip(avatar: const Icon(Icons.face, size: 18), label: Text('${p.name} · ${p.embeddings.length}'))]),
      ]);

  Widget _analysisTab() {
    final photos = assets.where((a) => a.type == AssetType.image).length;
    return ListView(children: [
      Row(children: [
        _stat('Fotos', photos),
        _stat('Videos', assets.length - photos),
        _stat('Copias', dups.fold(0, (a, g) => a + g.length - 1)),
      ]),
      const SizedBox(height: 12),
      const Text('Ubicación', style: TextStyle(fontWeight: FontWeight.bold)),
      for (final e in albums.entries) ListTile(dense: true, leading: const Icon(Icons.folder_outlined), title: Text(e.key), trailing: Text('${e.value}')),
      const Text('Duplicadas', style: TextStyle(fontWeight: FontWeight.bold)),
      for (final g in dups)
        Card(
            child: Column(children: [
          SizedBox(height: 90, child: ListView(scrollDirection: Axis.horizontal, children: [for (final a in g) Padding(padding: const EdgeInsets.all(4), child: AssetEntityThumb(a))])),
          Row(mainAxisAlignment: MainAxisAlignment.end, children: [
            TextButton(onPressed: () => _moveCopies(g), child: const Text('Mover')),
            TextButton(
                onPressed: () async {
                  await gallery.delete(g.sublist(1)); // conserva la primera
                  dups.remove(g);
                  setState(() {});
                },
                child: const Text('Borrar copias')),
          ])
        ])),
    ]);
  }

  Future<void> _moveCopies(List<AssetEntity> g) async {
    final paths = await PhotoManager.getAssetPathList(type: RequestType.common, onlyAll: false);
    if (!mounted) return;
    final target = await showDialog<AssetPathEntity>(
        context: context,
        builder: (_) => SimpleDialog(title: const Text('Mover a…'), children: [for (final p in paths) SimpleDialogOption(onPressed: () => Navigator.pop(context, p), child: Text(p.name))]));
    if (target == null) return;
    for (final a in g.sublist(1)) await gallery.moveTo(a, target);
    _msg('Copias movidas a ${target.name}');
  }

  Widget _stat(String l, int n) => Expanded(
      child: Card(child: Padding(padding: const EdgeInsets.all(12), child: Column(children: [Text('$n', style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold)), Text(l)]))));
}

class AssetEntityThumb extends StatelessWidget {
  final AssetEntity a;
  const AssetEntityThumb(this.a, {super.key});
  @override
  Widget build(BuildContext c) => FutureBuilder<Uint8List?>(
        future: a.thumbnailDataWithSize(const ThumbnailSize(200, 200)),
        builder: (_, s) => s.data == null ? const SizedBox(width: 82) : ClipRRect(borderRadius: BorderRadius.circular(10), child: Image.memory(s.data!, width: 82, height: 82, fit: BoxFit.cover)),
      );
}

/// Pantalla de confirmación: ¿es la misma persona? Sí / No
class ConfirmPage extends StatefulWidget {
  final List<AssetEntity> matches;
  const ConfirmPage(this.matches, {super.key});
  @override
  State<ConfirmPage> createState() => _ConfirmPageState();
}

class _ConfirmPageState extends State<ConfirmPage> {
  int i = 0;
  final yes = <AssetEntity>[];

  void answer(bool ok) {
    if (ok) yes.add(widget.matches[i]);
    if (i + 1 >= widget.matches.length) return Navigator.pop(context, yes);
    setState(() => i++);
  }

  @override
  Widget build(BuildContext context) {
    if (widget.matches.isEmpty) return Scaffold(appBar: AppBar(), body: const Center(child: Text('Sin coincidencias')));
    return Scaffold(
      appBar: AppBar(title: Text('Coincidencia ${i + 1} de ${widget.matches.length}')),
      body: Column(children: [
        Expanded(child: Padding(padding: const EdgeInsets.all(16), child: AssetEntityImage(widget.matches[i]))),
        const Text('¿Es la misma persona?', style: TextStyle(fontSize: 20)),
        Padding(
            padding: const EdgeInsets.all(16),
            child: Row(mainAxisAlignment: MainAxisAlignment.spaceEvenly, children: [
              FilledButton.icon(style: FilledButton.styleFrom(backgroundColor: Colors.red), onPressed: () => answer(false), icon: const Icon(Icons.close), label: const Text('No')),
              FilledButton.icon(style: FilledButton.styleFrom(backgroundColor: Colors.green), onPressed: () => answer(true), icon: const Icon(Icons.check), label: const Text('Sí')),
            ])),
      ]),
    );
  }
}

class AssetEntityImage extends StatelessWidget {
  final AssetEntity a;
  const AssetEntityImage(this.a, {super.key});
  @override
  Widget build(BuildContext c) => FutureBuilder<Uint8List?>(
        future: a.thumbnailDataWithSize(const ThumbnailSize(900, 900)),
        builder: (_, s) => s.data == null ? const Center(child: CircularProgressIndicator()) : ClipRRect(borderRadius: BorderRadius.circular(20), child: Image.memory(s.data!, fit: BoxFit.contain)),
      );
}
