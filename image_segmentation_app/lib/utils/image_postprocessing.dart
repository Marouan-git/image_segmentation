import 'dart:typed_data';
import 'dart:math';
import 'package:image/image.dart' as img;
import '../models/class_names.dart';
import 'package:image/src/font/arial_14.dart';

class ImagePostprocessing {
  static const int numClasses = 80; // Number of classes in COCO dataset
  static const double confThreshold = 0.25;
  static const double iouThreshold = 0.7;

  // Sigmoid function
  static double sigmoid(double x) {
    return 1 / (1 + exp(-x));
  }

  // Compute Intersection over Union (IoU) between two boxes
  static double computeIoU(List<double> boxA, List<double> boxB) {
    double xA = max(boxA[0], boxB[0]);
    double yA = max(boxA[1], boxB[1]);
    double xB = min(boxA[2], boxB[2]);
    double yB = min(boxA[3], boxB[3]);

    double interArea = max(0, xB - xA) * max(0, yB - yA);
    double boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
    double boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);

    double iou = interArea / (boxAArea + boxBArea - interArea);
    return iou;
  }

  // Non-Maximum Suppression
  static List<int> nms(List<List<double>> boxes, List<double> scores, double iouThreshold) {
    List<int> indices = List<int>.generate(scores.length, (index) => index);
    indices.sort((a, b) => scores[b].compareTo(scores[a]));

    List<int> keep = [];

    while (indices.isNotEmpty) {
      int current = indices.removeAt(0);
      keep.add(current);

      indices.removeWhere((index) {
        double iou = computeIoU(boxes[current], boxes[index]);
        return iou > iouThreshold;
      });
    }

    return keep;
  }

  // Postprocess the model outputs
  static Future<img.Image> postprocess(
    Map<String, dynamic> outputs,
    img.Image originalImage,
    double ratio,
    int padLeft,
    int padTop,
    List<List<int>> colors,
  ) async {
    const int inputWidth = 320;
    const int inputHeight = 320;

    // Extract predictions and maskProtos
    List<List<double>> predictions = outputs['predictions']; 
    List<List<List<double>>> maskProtos = outputs['maskProtos']; 

    int origHeight = originalImage.height;
    int origWidth = originalImage.width;

    int maskChannels = maskProtos.length;
    int maskHeight = maskProtos[0].length; 
    int maskWidth = maskProtos[0][0].length; 

    // Flatten maskProtos to [32][H*W]
    List<List<double>> maskProtosFlat = List.generate(
      maskChannels,
      (c) => maskProtos[c].expand((row) => row).toList(),
    );

    // Extract boxes, class scores, mask coefficients
    List<List<double>> boxes = [];
    List<double> scores = [];
    List<int> classIds = [];
    List<List<double>> maskCoeffs = [];

    for (int i = 0; i < predictions.length; i++) {
      List<double> pred = predictions[i];

      if (pred.length != 4 + numClasses + maskChannels) {
        // Unexpected pred length, skip
        continue;
      }

      double x = pred[0];
      double y = pred[1];
      double w = pred[2];
      double h = pred[3];

      // Class scores
      List<double> classScores = pred.sublist(4, 4 + numClasses);

      double maxScore = classScores.reduce(max);
      int classId = classScores.indexOf(maxScore);

      if (maxScore < confThreshold) {
        continue;
      }

      // // Define a list of target class names
      // List<String> targetClassNames = ['person', 'laptop', 'bottle', 'cell phone', 'chair', 'mouse']; // Adjust as needed

      // // Map class names to class IDs
      // List<int> targetClassIds = targetClassNames.map((name) => cocoClassNames.indexOf(name)).toList();

      // // **Check if the detected classId is in the list of targetClassIds**
      // if (!targetClassIds.contains(classId)) {
      //   continue; // Skip this detection
      // }

      // Mask coefficients
      List<double> maskCoeff = pred.sublist(4 + numClasses);

      if (maskCoeff.length != maskChannels) {
        print('Warning: maskCoeff length (${maskCoeff.length}) does not match maskChannels ($maskChannels)');
        continue;
      }

      boxes.add([x, y, w, h]);
      scores.add(maxScore);
      classIds.add(classId);
      maskCoeffs.add(maskCoeff);
    }

    if (boxes.isEmpty) {
      print('No detections after thresholding.');
      return originalImage;
    }

    // Adjust boxes to original image scale
    for (int i = 0; i < boxes.length; i++) {
      double x = boxes[i][0];
      double y = boxes[i][1];
      double w = boxes[i][2];
      double h = boxes[i][3];

      // Adjustments
      x = (x - padLeft) / ratio;
      y = (y - padTop) / ratio;
      w = w / ratio;
      h = h / ratio;

      // Convert to (x1, y1, x2, y2)
      double x1 = x - w / 2;
      double y1 = y - h / 2;
      double x2 = x + w / 2;
      double y2 = y + h / 2;

      // Clip to image boundaries
      x1 = x1.clamp(0, origWidth - 1);
      y1 = y1.clamp(0, origHeight - 1);
      x2 = x2.clamp(0, origWidth - 1);
      y2 = y2.clamp(0, origHeight - 1);

      boxes[i] = [x1, y1, x2, y2];
    }

    // Perform Non-Maximum Suppression
    List<int> keep = nms(boxes, scores, iouThreshold);

    // Keep only the detections after NMS
    List<List<double>> finalBoxes = [for (var idx in keep) boxes[idx]];
    List<int> finalClassIds = [for (var idx in keep) classIds[idx]];
    List<List<double>> finalMaskCoeffs = [for (var idx in keep) maskCoeffs[idx]];

    // Overlay masks on the original image
    img.Image annotatedImage = img.copyCrop(
      originalImage,
      x: 0,
      y: 0,
      width: origWidth,
      height: origHeight,
    );

    for (int i = 0; i < finalBoxes.length; i++) {
      List<double> maskCoeff = finalMaskCoeffs[i];
      int classId = finalClassIds[i];
      List<int> color = colors[classId % colors.length];

      // Compute maskFlat = sigmoid(maskCoeff * maskProtosFlat)
      List<double> maskFlat = List.generate(
        maskHeight * maskWidth,
        (idx) {
          double m = 0.0;
          for (int c = 0; c < maskChannels; c++) {
            m += maskCoeff[c] * maskProtosFlat[c][idx];
          }
          return sigmoid(m);
        },
      );

      // Reshape maskFlat to [maskHeight][maskWidth]
      List<List<double>> mask = List.generate(
        maskHeight,
        (h) => maskFlat.sublist(h * maskWidth, (h + 1) * maskWidth),
      );

      // Resize mask to inputWidth x inputHeight
      img.Image maskImage = img.Image(width: maskWidth, height: maskHeight);
      for (int y = 0; y < maskHeight; y++) {
        for (int x = 0; x < maskWidth; x++) {
          int value = (mask[y][x] * 255).toInt().clamp(0, 255);
          maskImage.setPixelRgba(x, y, value, value, value, 255);
        }
      }

      img.Image resizedMask = img.copyResize(
        maskImage,
        width: inputWidth,
        height: inputHeight,
        interpolation: img.Interpolation.linear,
      );

      // Remove padding from masks
      int dhInt = padTop;
      int dwInt = padLeft;
      int croppedWidth = inputWidth - dwInt * 2;
      int croppedHeight = inputHeight - dhInt * 2;

      img.Image croppedMask = img.copyCrop(
        resizedMask,
        x: dwInt,
        y: dhInt,
        width: croppedWidth,
        height: croppedHeight,
      );

      // Resize masks to original image size
      img.Image finalMask = img.copyResize(
        croppedMask,
        width: origWidth,
        height: origHeight,
        interpolation: img.Interpolation.linear,
      );

      // Apply threshold to get binary mask
      double maskThreshold = 0.5;
      for (int y = 0; y < origHeight; y++) {
        for (int x = 0; x < origWidth; x++) {
          int maskValue = finalMask.getPixel(x, y).r.toInt() & 0xFF;
          if (maskValue > (maskThreshold * 255)) {
            img.Pixel origPixel = annotatedImage.getPixel(x, y);
            int r = origPixel.r.toInt();
            int g = origPixel.g.toInt();
            int b = origPixel.b.toInt();

            // Blend the mask color with the original pixel
            r = ((r * 0.5) + (color[0] * 0.5)).clamp(0, 255).toInt();
            g = ((g * 0.5) + (color[1] * 0.5)).clamp(0, 255).toInt();
            b = ((b * 0.5) + (color[2] * 0.5)).clamp(0, 255).toInt();

            annotatedImage.setPixelRgba(x, y, r, g, b, 255);
          }
        }
      }

      // Draw bounding box
      List<double> box = finalBoxes[i];
      int x1 = box[0].toInt();
      int y1 = box[1].toInt();
      int x2 = box[2].toInt();
      int y2 = box[3].toInt();

      img.drawRect(
        annotatedImage,
        x1: x1,
        y1: y1,
        x2: x2,
        y2: y2,
        color: img.ColorRgb8(color[0], color[1], color[2]),
      );

      // Draw class label
      String className = cocoClassNames[classId];

      // Choose a font
      img.BitmapFont font = img.arial14;

      // Position the text above the bounding box
      int textX = x1;
      int textY = y1 - font.lineHeight;

      // Ensure the text doesn't go off the top of the image
      if (textY < 0) {
        textY = y1 + font.lineHeight;
      }

      // Draw the class name
      img.drawString(
        annotatedImage,
        className,
        font: font,
        x: textX,
        y: textY,
        color: img.ColorRgb8(color[0], color[1], color[2]),
      );
    }

    return annotatedImage;
  }
}
