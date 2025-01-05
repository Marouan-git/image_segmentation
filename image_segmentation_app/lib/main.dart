import 'package:flutter/material.dart';
import 'pages/home_page.dart';

void main() {
  runApp(ImageSegmentationApp());
}

class ImageSegmentationApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Image Segmentation App',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: HomePage(),
    );
  }
}