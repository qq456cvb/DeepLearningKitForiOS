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

public extension DeepNetwork {
    // e.g. 32x32x3 for CIFAR-10/100
    // as [1.0, 3.0, 32.0, 32.0]
    
    func iou(box1: CGRect, box2 : CGRect) -> Float {
        let tb = min(box1.maxX, box2.maxX) - max(box1.minX, box2.minX)
        let lr = min(box1.maxY, box2.maxY) - max(box1.minY, box2.minY)
        if tb < 0 || lr < 0 {
            return 0.0
        }
        let intersection = tb * lr
        return Float(intersection) / Float(box1.width * box1.height + box2.width * box2.height - intersection)
    }
    
    public func yoloDetect(_ flattenedTensorWithImage: [Float], imageView: UIImageView) {
        let start = Date()
        
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
        var candidateBox = [CGRect]()
        var probs = [Float]()
        let image = imageView.image!
        
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
                        
                        let box_x = (output[1078 + x * 7 * 2 * 4 + y * 2 * 4 + i * 4] + Float(x)) / 7.0 * Float(image.size.width)
                        let box_y = (output[1078 + x * 7 * 2 * 4 + y * 2 * 4 + i * 4 + 1] + Float(y)) / 7.0 * Float(image.size.height)
                        print(output[1078 + x * 7 * 2 * 4 + y * 2 * 4 + i * 4 + 2])
                        print(output[1078 + x * 7 * 2 * 4 + y * 2 * 4 + i * 4 + 3])
                        let box_width = pow(output[1078 + x * 7 * 2 * 4 + y * 2 * 4 + i * 4 + 2], 2) * Float(image.size.width)
                        let box_height = pow(output[1078 + x * 7 * 2 * 4 + y * 2 * 4 + i * 4 + 3], 2) * Float(image.size.height)
                        
                        print(box_x)
                        print(box_y)
                        print(box_width)
                        print(box_height)
                        candidateBox.append(CGRect(x: CGFloat(box_x - box_width / 2), y: CGFloat(box_y - box_height / 2), width: CGFloat(box_width), height: CGFloat(box_height)))

                    }
                }
            }
        }
        
        if candidateBox.count == 0 {
            return
        }
        
        candidateBox = candidateBox.sorted(by: { (rect1:CGRect, rect2:CGRect) -> Bool in
            let idx1 = candidateBox.index(of: rect1)
            let idx2 = candidateBox.index(of: rect2)
            return probs[idx1!] > probs[idx2!]
        })
        
        for i in 0...(candidateBox.count-1) {
            if (probs[i] == 0.0 || i == candidateBox.count-1) {
                continue
            }
            for j in (i+1)...(candidateBox.count-1) {
                if iou(box1: candidateBox[i], box2: candidateBox[j]) > iou_threshold {
//                    probs[j] = 0.0
                }
            }
            UIGraphicsBeginImageContextWithOptions(image.size, true, 0)
            image.draw(at: CGPoint(x: 0, y: 0))
            let p = UIBezierPath()
            p.move(to: CGPoint(x: candidateBox[i].minX, y: candidateBox[i].minY))
            p.addLine(to: CGPoint(x: candidateBox[i].width, y: candidateBox[i].height))
            UIColor.red.set()
            p.lineWidth = 10.0
            p.stroke()
            p.fill()
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
        
        let (lastLayerName, lastMetalBuffer) = namedDataLayers.last!
        NSLog(lastLayerName)
        // modified
        let data = Data(bytesNoCopy: UnsafeMutableRawPointer(lastMetalBuffer.contents()),
            count: output.count*4, deallocator: .none)
        (data as NSData).getBytes(&output, length:(Int(output.count)) * 4)
        print(output)
        
        let maxValue = output.max()
        let indexOfMaxValue = Float(output.index(of: maxValue!)!)
        
        print("maxValue = \(maxValue), indexofMaxValue = \(indexOfMaxValue)")
        
        // empty command buffers!
        gpuCommandLayers = []
        
        // return index
        return Float(output.index(of: output.max()!)!)
    }

}
