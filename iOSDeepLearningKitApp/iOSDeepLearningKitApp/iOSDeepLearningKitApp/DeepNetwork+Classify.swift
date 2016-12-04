//
//  DeepNetwork+Classify.swift
//  MemkiteMetal
//
//  Created by Amund Tveit on 25/11/15.
//  Copyright © 2015 memkite. All rights reserved.
//

import Foundation
import Metal
import UIKit

class Prediction {
    var rect = CGRect()
    var cls = String()
    var prob:Float = 0.0
}

public extension DeepNetwork {
    // e.g. 32x32x3 for CIFAR-10/100
    // as [1.0, 3.0, 32.0, 32.0]
    
    func iou(box1: CGRect, box2 : CGRect) -> Float {
        let tb = min(box1.origin.x + box1.size.width, box2.origin.x + box2.size.width) - max(box1.origin.x, box2.origin.x)
        let lr = min(box1.origin.y + box1.size.height, box2.origin.y + box2.size.height) - max(box1.origin.y, box2.origin.y)
        if tb < 0 || lr < 0 {
            return 0.0
        }
        let intersection = tb * lr
        return Float(intersection) / Float(box1.size.width * box1.size.height + box2.size.width * box2.size.height - intersection)
    }
    
    public func yoloDetect(_ image: UIImage, imageView: UIImageView) {
        let start = Date()
        
//        let inputTensor = createMetalBuffer(flattenedTensorWithImage, metalDevice:metalDevice)
        
//        flattenedTensorWithImage.
//        namedDataLayers[0].1.contents().copyBytes(from: flattenedTensorWithImage, count: flattenedTensorWithImage.count * 4)
        
        for commandBuffer in gpuCommandLayers {
            commandBuffer.commit()
        }
        
        
        // wait until last layer in conv.net is finished
        gpuCommandLayers.last?.waitUntilCompleted()
        
        print("Time to run network: \(Date().timeIntervalSince(start))")
        
        
        // TODO: fix hardcoding better..
        var output =  [Float](repeating: 0, count: 1470)
        
        let (lastLayerName, lastMetalBuffer) = namedDataLayers.last!
        NSLog(lastLayerName)
        // modified
        let data = Data(bytesNoCopy: UnsafeMutableRawPointer(lastMetalBuffer.contents()),
                        count: output.count*4, deallocator: .none)
        (data as NSData).getBytes(&output, length:(Int(output.count)) * 4)
//        print(output)
        
//        let maxValue = output.max()
//        let indexOfMaxValue = Float(output.index(of: maxValue!)!)
        
//        print("maxValue = \(maxValue), indexofMaxValue = \(indexOfMaxValue)")
        let classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                       "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        // empty command buffers!
        gpuCommandLayers = []
        
        let threshold: Float = 0.2
        let iou_threshold: Float = 0.5
        var predictions = [Prediction]()
        var probs = [Float]()
        
        print("image size: \(image.size.width) * \(image.size.height)")
        for x in 0...6 {
            for y in 0...6 {
                for i in 0...1 {
                    var max: Float = 0.0
                    var max_class: Int = 0
                    let confidence = output[980 + x * 7 * 2 + y * 2 + i]
                    for j in 0...19 {
                        let index = x * 7 * 20 + y * 20 + j
                        let class_prob = output[index]
                        if class_prob * confidence > max {
                            max = class_prob * confidence
                            max_class = j
                        }
                    }
                    
                    if max > threshold {
                        print("find one class: \(classes[max_class])")
                        probs.append(max)
                        
                        // x, y swapped!
                        var box_x = (output[1078 + x * 7 * 2 * 4 + y * 2 * 4 + i * 4] + Float(y)) / 7.0 * Float(image.size.width)
                        var box_y = (output[1078 + x * 7 * 2 * 4 + y * 2 * 4 + i * 4 + 1] + Float(x)) / 7.0 * Float(image.size.height)
                        print(output[1078 + x * 7 * 2 * 4 + y * 2 * 4 + i * 4 + 0])
                        print(output[1078 + x * 7 * 2 * 4 + y * 2 * 4 + i * 4 + 1])
                        print(output[1078 + x * 7 * 2 * 4 + y * 2 * 4 + i * 4 + 2])
                        print(output[1078 + x * 7 * 2 * 4 + y * 2 * 4 + i * 4 + 3])
                        var box_width = pow(output[1078 + x * 7 * 2 * 4 + y * 2 * 4 + i * 4 + 2], 2) * Float(image.size.width)
                        var box_height = pow(output[1078 + x * 7 * 2 * 4 + y * 2 * 4 + i * 4 + 3], 2) * Float(image.size.height)
                        
//                        print(box_x)
//                        print(box_y)
//                        print(box_width)
//                        print(box_height)
                        
                        box_x -= box_width / 2
                        box_y -= box_height / 2
                        if box_x < 0 {
                            box_x = 0
                        }
                        if box_y < 0 {
                            box_y = 0
                        }
                        if box_x + box_width >= Float(image.size.width) {
                            box_width = Float(image.size.width) - box_x - 1
                        }
                        if box_y + box_height >= Float(image.size.height) {
                            box_height = Float(image.size.height) - box_y - 1
                        }
                        
                        let pred = Prediction()
                        pred.rect = CGRect(x: CGFloat(box_x), y: CGFloat(box_y), width: CGFloat(box_width), height: CGFloat(box_height))
                        pred.cls = classes[max_class]
                        pred.prob = max
                        predictions.append(pred)
                    }
                }
            }
        }
        
        if predictions.count == 0 {
            return
        }
        
        predictions = predictions.sorted(by: {
            $0.prob > $1.prob
        })
        
        DispatchQueue.main.async {
            UIGraphicsBeginImageContextWithOptions(image.size, true, 0)
            image.draw(at: CGPoint(x: 0, y: 0))
            UIColor.red.set()
            for i in 0...(predictions.count-1) {
                if (probs[i] == 0.0) {
                    continue
                }
                if i+1 <= predictions.count-1 {
                    for j in (i+1)...(predictions.count-1) {
                        if self.iou(box1: predictions[i].rect, box2: predictions[j].rect) > iou_threshold {
                            probs[j] = 0.0
                        }
                    }
                }
                
                // set the text color to dark gray
                let fieldColor: UIColor = UIColor.red
                
                // set the font to Helvetica Neue 18
                let fieldFont = UIFont(name: "Helvetica Neue", size: image.size.width / 30.0)
                
                // set the line spacing to 6
                let paraStyle = NSMutableParagraphStyle()
                paraStyle.lineSpacing = 6.0
                
                // set the Obliqueness to 0.1
                let skew = 0.1
                
                let attributes: NSDictionary = [
                    NSForegroundColorAttributeName: fieldColor,
                    NSParagraphStyleAttributeName: paraStyle,
                    NSObliquenessAttributeName: skew,
                    NSFontAttributeName: fieldFont!
                ]
                
                predictions[i].cls.draw(in: CGRect(x:predictions[i].rect.origin.x + image.size.width / 100.0, y:predictions[i].rect.origin.y, width: predictions[i].rect.width, height: image.size.width / 30.0), withAttributes: attributes as? [String : Any])
                
                let p = UIBezierPath()
                p.move(to: CGPoint(x: predictions[i].rect.origin.x, y: predictions[i].rect.origin.y))
                p.addLine(to: CGPoint(x: predictions[i].rect.origin.x + predictions[i].rect.size.width, y: predictions[i].rect.origin.y))
                p.move(to: CGPoint(x: predictions[i].rect.origin.x + predictions[i].rect.size.width, y: predictions[i].rect.origin.y))
                p.addLine(to: CGPoint(x: predictions[i].rect.origin.x + predictions[i].rect.size.width, y: predictions[i].rect.origin.y + predictions[i].rect.size.height))
                p.move(to: CGPoint(x: predictions[i].rect.origin.x + predictions[i].rect.size.width, y: predictions[i].rect.origin.y + predictions[i].rect.size.height))
                p.addLine(to: CGPoint(x: predictions[i].rect.origin.x, y: predictions[i].rect.origin.y + predictions[i].rect.size.height))
                p.move(to: CGPoint(x: predictions[i].rect.origin.x, y: predictions[i].rect.origin.y + predictions[i].rect.size.height))
                p.addLine(to: CGPoint(x: predictions[i].rect.origin.x, y: predictions[i].rect.origin.y))
                
                p.lineWidth = image.size.width / 100.0
                p.stroke()
                p.fill()
                
            }
            
            
            imageView.image = UIGraphicsGetImageFromCurrentImageContext()
            
            UIGraphicsEndImageContext()
        }
        // return index
//        return Float(output.index(of: output.max()!)!)
    }
    public func classify(_ flattenedTensorWithImage: [Float]) -> Float {
        let start = Date()
        
        // from 2.2 in http://memkite.com/blog/2014/12/30/example-of-sharing-memory-between-gpu-and-cpu-with-swift-and-metal-for-ios8/
        /*let xvectorVoidPtr = COpaquePointer(imageBuffer.contents())
        let xvectorFloatPtr = UnsafeMutablePointer<Float>(xvectorVoidPtr)
        let xvectorFloatBufferPtr = UnsafeMutableBufferPointer<Float>(start:xvectorFloatPtr, count: flattenedTensorWithImage.count)
        for index in xvectorFloatBufferPtr.startIndex..<xvectorFloatBufferPtr.endIndex {
            xvectorFloatBufferPtr[index] = Float(flattenedTensorWithImage[index])
        }
*/
        
        
        
        for commandBuffer in gpuCommandLayers {
            commandBuffer.commit()
        }
        
        
        // wait until last layer in conv.net is finished
        gpuCommandLayers.last?.waitUntilCompleted()
        
        print("Time to run network: \(Date().timeIntervalSince(start))")
        
        
        // TODO: fix hardcoding better..
        var output =  [Float](repeating: 0, count: 1470)
        
        print(namedDataLayers.count)
        let (lastLayerName, lastMetalBuffer) = namedDataLayers[28]
        NSLog(lastLayerName)
        // modified
        let data = Data(bytesNoCopy: UnsafeMutableRawPointer(lastMetalBuffer.contents()),
            count: output.count*4, deallocator: .none)
        (data as NSData).getBytes(&output, length:(Int(output.count)) * 4)
//        print(output)
        print(output.reduce(0, +))
        
        let maxValue = output.max()
        let indexOfMaxValue = Float(output.index(of: maxValue!)!)
        
        print("maxValue = \(maxValue), indexofMaxValue = \(indexOfMaxValue)")
        
        // empty command buffers!
        gpuCommandLayers = []
        
        // return index
        return Float(output.index(of: output.max()!)!)
    }

}
