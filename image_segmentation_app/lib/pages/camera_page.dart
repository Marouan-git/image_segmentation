import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import '../services/onnx_service.dart';
import '../utils/image_preprocessing.dart';
import '../utils/image_postprocessing.dart';
import '../models/class_names.dart';
import 'package:image/image.dart' as img;

class CameraPage extends StatefulWidget {
  @override
  _CameraPageState createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  CameraController? _controller;
  bool _isCameraInitialized = false;
  List<CameraDescription>? cameras;
  OnnxService onnxService = OnnxService();
  bool isProcessing = false;
  ui.Image? annotatedImage;
  List<List<int>> colors = [];

  int inferenceTime = 0; // In milliseconds
  int totalTime = 0;     // Total processing time in milliseconds

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _initializeModel();
    _generateColors();
  }

  Future<void> _initializeModel() async {
    await onnxService.initializeModel();
  }

  void _generateColors() {
    // Generate random colors for each class
    for (int i = 0; i < cocoClassNames.length; i++) {
      colors.add([
        (255 * (i / cocoClassNames.length)).toInt(),
        (255 * ((i * 3) % cocoClassNames.length) / cocoClassNames.length).toInt(),
        (255 * ((i * 7) % cocoClassNames.length) / cocoClassNames.length).toInt(),
      ]);
    }
  }

  Future<void> _initializeCamera() async {
    // Request camera permission
    final status = await Permission.camera.request();

    if (status.isGranted) {
      cameras = await availableCameras();
      if (cameras != null && cameras!.isNotEmpty) {
        _controller = CameraController(
          cameras![0],
          ResolutionPreset.medium,
          enableAudio: false,
          imageFormatGroup: ImageFormatGroup.yuv420,
        );

        await _controller!.initialize();
        if (!mounted) return;
        setState(() {
          _isCameraInitialized = true;
        });

        // Start image stream
        _controller!.startImageStream(_processCameraImage);
      }
    } else {
      // Handle the case when the user denies the permission
      print('Camera permission denied');
    }
  }

  void _processCameraImage(CameraImage cameraImage) {
    if (isProcessing || onnxService == null) return;

    isProcessing = true;

    _runInference(cameraImage).then((Map<String, dynamic>? result) {
      if (result != null) {
        setState(() {
          annotatedImage = result['annotatedImage'];
          inferenceTime = result['inferenceTime'];
          totalTime = result['totalTime'];
        });
      }
      isProcessing = false;
    });
  }

  Future<Map<String, dynamic>?> _runInference(CameraImage cameraImage) async {
    try {
      // Start total processing time stopwatch
      final totalStopwatch = Stopwatch()..start();

      // Convert CameraImage to img.Image
      img.Image? image = await convertYUV420toImage(cameraImage);
      if (image == null) {
        throw Exception('Failed to convert camera image to RGB image');
      }

      // Preprocess the image
      Float32List inputImage = await ImagePreprocessing.preprocessImage(image);

      // Start inference time stopwatch
      final inferenceStopwatch = Stopwatch()..start();

      // Run inference
      Map<String, dynamic> outputs = await onnxService.runInference(inputImage);

      // Stop inference stopwatch
      inferenceStopwatch.stop();
      final int inferenceTimeMs = inferenceStopwatch.elapsedMilliseconds;

      // Postprocess the outputs
      img.Image annotatedImg = await ImagePostprocessing.postprocess(
        outputs,
        image,
        ImagePreprocessing.ratio,
        ImagePreprocessing.padLeft,
        ImagePreprocessing.padTop,
        colors,
      );

      // Convert img.Image to ui.Image for display
      ui.Codec codec = await ui.instantiateImageCodec(Uint8List.fromList(img.encodeJpg(annotatedImg)));
      ui.FrameInfo frameInfo = await codec.getNextFrame();
      ui.Image uiImage = frameInfo.image;

      // Stop total stopwatch
      totalStopwatch.stop();
      final int totalTimeMs = totalStopwatch.elapsedMilliseconds;

      // Return both the annotated image and the inference time
      return {
        'annotatedImage': uiImage,
        'inferenceTime': inferenceTimeMs,
        'totalTime': totalTimeMs,
      };
    } catch (e) {
      print('Error processing image: $e');
      return null;
    }
  }

  Future<img.Image?> convertYUV420toImage(CameraImage image) async {
    try {
      final int width = image.width;
      final int height = image.height;

      final int uvRowStride = image.planes[1].bytesPerRow;
      final int uvPixelStride = image.planes[1].bytesPerPixel ?? 1;

      // Create an empty image buffer
      img.Image imgData = img.Image(width: width, height: height);

      for (int y = 0; y < height; y++) {
        final int uvRow = uvRowStride * (y ~/ 2);
        final int yRow = image.planes[0].bytesPerRow * y;
        for (int x = 0; x < width; x++) {
          final int uvCol = uvPixelStride * (x ~/ 2);

          final int yPixel = image.planes[0].bytes[yRow + x];
          final int uPixel = image.planes[1].bytes[uvRow + uvCol];
          final int vPixel = image.planes[2].bytes[uvRow + uvCol];

          // YUV to RGB conversion
          final double yValue = yPixel.toDouble();
          final double uValue = uPixel.toDouble() - 128.0;
          final double vValue = vPixel.toDouble() - 128.0;

          double r = yValue + 1.402 * vValue;
          double g = yValue - 0.344136 * uValue - 0.714136 * vValue;
          double b = yValue + 1.772 * uValue;

          // Clamp values to [0, 255]
          r = r.clamp(0, 255);
          g = g.clamp(0, 255);
          b = b.clamp(0, 255);

          imgData.setPixelRgba(x, y, r.toInt(), g.toInt(), b.toInt(), 255);
        }
      }

      // Rotate the image by 90 degrees clockwise
      img.Image rotatedImage = img.copyRotate(imgData, angle: 90);

      return rotatedImage;
    } catch (e) {
      print("Error converting YUV420 to RGB: $e");
      return null;
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  void _stopSegmentation() {
    _controller?.stopImageStream();
    Navigator.pop(context);
  }

  @override
  Widget build(BuildContext context) {
    if (!_isCameraInitialized || _controller == null) {
      return Scaffold(
        appBar: AppBar(
          title: Text('Camera'),
        ),
        body: Center(child: CircularProgressIndicator()),
      );
    }

    // Get the aspect ratio of the camera
    double cameraAspectRatio = _controller!.value.aspectRatio;

    return Scaffold(
      body: Column(
        children: [
          // Camera preview and overlay
          Expanded(
            child: Stack(
              children: [
                AspectRatio(
                  aspectRatio: cameraAspectRatio,
                  child: CameraPreview(_controller!),
                ),
                if (annotatedImage != null)
                  AspectRatio(
                    aspectRatio: cameraAspectRatio,
                    child: CustomPaint(
                      painter: ImagePainter(annotatedImage!),
                    ),
                  ),
              ],
            ),
          ),
          // Button and texts below the camera preview
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Column(
              children: [
                ElevatedButton(
                  onPressed: _stopSegmentation,
                  child: Text('Stop Segmentation'),
                ),
                SizedBox(height: 10),
                Text(
                  'Inference Time: $inferenceTime ms',
                  style: TextStyle(
                    color: Colors.black,
                    fontSize: 18,
                  ),
                ),
                SizedBox(height: 5),
                Text(
                  'Total Time: $totalTime ms',
                  style: TextStyle(
                    color: Colors.black,
                    fontSize: 18,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class ImagePainter extends CustomPainter {
  final ui.Image image;

  ImagePainter(this.image);

  @override
  void paint(Canvas canvas, Size size) {
    Paint paint = Paint();
    canvas.drawImage(image, Offset.zero, paint);
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => true;
}
