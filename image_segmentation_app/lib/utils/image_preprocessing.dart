import 'dart:math';
import 'package:image/image.dart' as img;
import 'dart:typed_data';



class ImagePreprocessing {
  static const int inputHeight = 320;
  static const int inputWidth = 320;
  static const List<double> mean = [0.0, 0.0, 0.0]; // YOLOv8 does not use mean normalization
  static const List<double> std = [1.0, 1.0, 1.0]; // YOLOv8 does not use std normalization
  static double ratio = 1.0;
  static int padLeft = 0;
  static int padTop = 0;

  //Letterbox function to resize and pad the image
  // static img.Image letterbox(img.Image image, int newWidth, int newHeight) {
  //   int width = image.width;
  //   int height = image.height;

  //   //double scale = (newWidth / width).clamp(0.0, newHeight / height);

  //   double r = min(newWidth / width, newHeight / height);
  //   ratio = r;

  //   int resizedWidth = (width * r).round();
  //   int resizedHeight = (height * r).round();

  //   // int resizedWidth = (width * scale).round();
  //   // int resizedHeight = (height * scale).round();

  //   img.Image resizedImage = img.copyResize(image, width: resizedWidth, height: resizedHeight);

  //   int padWidth = newWidth - resizedWidth;
  //   int padHeight = newHeight - resizedHeight;

  //   int padLeft = padWidth ~/ 2;
  //   int padTop = padHeight ~/ 2;

  //   img.Image paddedImage = img.copyResize(resizedImage, width: newWidth, height: newHeight, interpolation: img.Interpolation.nearest);

  //   return paddedImage;
  // }
  static img.Image letterbox(img.Image image, int newWidth, int newHeight) {
    int width = image.width;
    int height = image.height;

    double r = min(newWidth / width, newHeight / height);
    ratio = r;

    int resizedWidth = (width * r).round();
    int resizedHeight = (height * r).round();

    // Resize the image
    img.Image resizedImage = img.copyResize(image, width: resizedWidth, height: resizedHeight);

    int padWidth = newWidth - resizedWidth;
    int padHeight = newHeight - resizedHeight;

    padLeft = padWidth ~/ 2;
    padTop = padHeight ~/ 2;

    // Assign padLeft and padTop to class variables
    ImagePreprocessing.padLeft = padLeft;
    ImagePreprocessing.padTop = padTop;

    // Create new image and fill with padding color (114, 114, 114)
    img.Image paddedImage = img.Image(width: newWidth, height: newHeight);
    // Fill the image with gray color
    for (int y = 0; y < newHeight; y++) {
      for (int x = 0; x < newWidth; x++) {
        paddedImage.setPixelRgba(x, y, 114, 114, 114, 255);  // RGB(114,114,114)
      }
    }

    // Copy resized image into padded image
    for (int y = 0; y < resizedImage.height; y++) {
      for (int x = 0; x < resizedImage.width; x++) {
        var pixel = resizedImage.getPixel(x, y);
        paddedImage.setPixelRgba(x + padLeft, y + padTop, pixel.r, pixel.g, pixel.b, pixel.a);
      }
    }

    return paddedImage;
  }



//   static Future<Float32List> preprocessImage(img.Image image) async {
//   // The image is already decoded

//   // Resize and pad the image
//   img.Image inputImage = letterbox(image, inputWidth, inputHeight);

//   // Convert image to Float32List
//   List<double> imageData = [];

//   for (int y = 0; y < inputHeight; y++) {
//     for (int x = 0; x < inputWidth; x++) {
//       img.Pixel pixel = inputImage.getPixel(x, y);
//       double r = (((pixel.r as int) >> 16) & 0xFF) / 255.0;
//       double g = (((pixel.g as int) >> 8) & 0xFF) / 255.0;
//       double b = ((pixel.b as int) & 0xFF) / 255.0;

//       // Apply normalization if necessary
//       imageData.add((r - mean[0]) / std[0]);
//       imageData.add((g - mean[1]) / std[1]);
//       imageData.add((b - mean[2]) / std[2]);
//     }
//   }

//   // Convert List<double> to Float32List
//   Float32List inputTensor = Float32List.fromList(imageData);

//   return inputTensor;
// }
static Future<Float32List> preprocessImage(img.Image image) async {
  // Resize and pad the image
  img.Image inputImage = letterbox(image, inputWidth, inputHeight);

  // Convert image to Float32List in CHW format
  List<double> imageData = [];

  for (int c = 0; c < 3; c++) { // Channels first
    for (int y = 0; y < inputHeight; y++) {
      for (int x = 0; x < inputWidth; x++) {
        img.Pixel pixel = inputImage.getPixel(x, y);

        double value;
        if (c == 0) {
          value = pixel.r / 255.0; // Red channel
        } else if (c == 1) {
          value = pixel.g / 255.0; // Green channel
        } else {
          value = pixel.b / 255.0; // Blue channel
        }

        imageData.add(value);
      }
    }
  }

  // Convert List<double> to Float32List
  Float32List inputTensor = Float32List.fromList(imageData);

  return inputTensor;
}
}
