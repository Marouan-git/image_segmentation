import 'package:flutter/material.dart';
import 'camera_page.dart';

class HomePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    void _launchSegmentation() {
      Navigator.push(
        context,
        MaterialPageRoute(builder: (context) => CameraPage()),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: Text('Image Segmentation App'),
      ),
      body: Center(
        child: ElevatedButton(
          onPressed: _launchSegmentation,
          child: Text('Launch Image Segmentation'),
        ),
      ),
    );
  }
}
