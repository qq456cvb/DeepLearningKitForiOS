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
        gpuCommandLayers.last!.waitUntilCompleted()
        
        print("Time to run network: \(Date().timeIntervalSince(start))")
        
        
        // TODO: fix hardcoding better..
        var output =  [Float](repeating: 0.0, count: 1470)
        
        let (lastLayerName, lastMetalBuffer) = namedDataLayers.last!
        NSLog(lastLayerName)
        // modified
        let data = Data(bytesNoCopy: UnsafeMutableRawPointer(lastMetalBuffer.contents()),
            count: output.count*MemoryLayout<Float>.size, deallocator: .none)
        (data as NSData).getBytes(&output, length:(Int(output.count)) * MemoryLayout<Float>.size)
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
