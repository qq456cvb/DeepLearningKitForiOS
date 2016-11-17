//
//  DeepNetwork+Classify.swift
//  MemkiteMetal
//
//  Created by Amund Tveit on 25/11/15.
//  Copyright © 2015 memkite. All rights reserved.
//

import Foundation
import Metal

public extension DeepNetwork {
    // e.g. 32x32x3 for CIFAR-10/100
    // as [1.0, 3.0, 32.0, 32.0]
    
    public func yoloDetect(_ flattenedTensorWithImage: [Float]) {
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
                    }
                }
            }
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
