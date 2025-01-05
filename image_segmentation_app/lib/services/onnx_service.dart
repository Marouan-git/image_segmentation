import 'dart:isolate';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';

class OnnxService {
  // Singleton instance
  static final OnnxService _instance = OnnxService._internal();

  // Loaded ONNX session (kept in memory)
  late OrtSession _session;

  // Private constructor (singleton)
  OnnxService._internal();

  // Factory constructor to return the singleton instance
  factory OnnxService() {
    return _instance;
  }

  // Initialize the ONNX model (this should be called only once)
  Future<void> initializeModel() async {
    print('Initializing ONNX model...');

    try {
      // Initialize the ONNX Runtime environment
      OrtEnv.instance.init();

      // Load the model from assets
      const modelFileName = 'assets/models/yolo11n-seg.onnx';
      final rawAssetFile = await rootBundle.load(modelFileName);
      final modelBytes = rawAssetFile.buffer.asUint8List();

      // Create session options
      final sessionOptions = OrtSessionOptions();

      // Create and store the ONNX session in memory
      _session = OrtSession.fromBuffer(modelBytes, sessionOptions);
      print('ONNX model initialized successfully.');
    } catch (e) {
      print('Error initializing ONNX model: $e');
      throw Exception('Failed to initialize ONNX model');
    }
  }

  Future<Map<String, dynamic>> runInference(Float32List inputImage) async {
  try {
    print('Running ONNX inference...');

    // Prepare input tensor for the image (1, 3, inputHeight, inputWidth)
    final inputTensor = OrtValueTensor.createTensorWithDataList(
      inputImage,
      [1, 3, 320, 320], // Adjust input size if different
    );

    // Define the input map with tensor names and corresponding tensor values
    final inputs = {
      'images': inputTensor, // Adjust input name if different
    };

    // Create OrtRunOptions object
    final runOptions = OrtRunOptions();

    print('Running inference...');

    // Run the model inference
    final outputs = _session.run(
      runOptions,
      inputs,
    );

    print('Inference completed.');

    // Release the input tensors and run options after the inference
    inputTensor.release();
    runOptions.release();

    // Extract outputs
    final predictionsValue = outputs[0]?.value as List<List<List<double>>>?;
    final maskProtosValue = outputs[1]?.value as List<List<List<List<double>>>>?;

    if (predictionsValue == null || maskProtosValue == null) {
      throw Exception('Failed to get outputs from the model.');
    }

    // Since batch size is 1, we can ignore the outermost list
    List<List<double>> predictionsData = predictionsValue[0]; // Shape: [116][8400]

    // Transpose predictionsData to shape [8400][116]
    int numFeatures = predictionsData.length; // Should be 116
    int numAnchors = predictionsData[0].length; // Should be 8400

    print('Number of features (per detection): $numFeatures');
    print('Number of anchors/detections: $numAnchors');

    // Transpose predictionsData
    List<List<double>> predictions = List.generate(numAnchors, (i) {
      return List.generate(numFeatures, (j) {
        return predictionsData[j][i];
      });
    });

    // Process maskProtos
    List<List<List<double>>> maskProtos = maskProtosValue[0]; 

    // Return the processed outputs
    return {
      'predictions': predictions, 
      'maskProtos': maskProtos,   
    };
  } catch (e) {
    print('Error during ONNX inference: $e');
    throw Exception('Inference failed');
  }
}

}