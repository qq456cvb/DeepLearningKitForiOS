//
//  RectifierLayer.swift
//  MemkiteMetal
//
//  Created by Amund Tveit on 25/11/15.
//  Copyright © 2015 memkite. All rights reserved.
//

import Foundation
import Metal

func createRectifierLayer(_ layer:NSDictionary, inputBuffer: MTLBuffer, metalCommandQueue: MTLCommandQueue, metalDefaultLibrary:MTLLibrary, metalDevice:MTLDevice) -> (MTLBuffer,MTLCommandBuffer) {
    print(" ==> createrectifierlayer")
//    let metalCommandBuffer = metalCommandQueue.commandBuffer()
    let metalCommandBuffer = metalCommandQueue.makeCommandBufferWithUnretainedReferences()

    var params = NSDictionary()
    var negativeSlope:Float = 0.0
    
    if let relu_param = layer["relu_param"] {
        params = relu_param as! NSDictionary
        if let val = params["negative_slope"] {
            negativeSlope = val as! Float
        }
    }

    let result = addRectifierCommandToCommandBuffer(metalCommandBuffer, negativeSlope: negativeSlope, inputBuffer: inputBuffer,
        metalDefaultLibrary: metalDefaultLibrary, metalDevice:metalDevice)
    //metalCommandBuffer.commit()
    
    print(" <== createrectifierlayer")
    return (result, metalCommandBuffer)
}


func addRectifierCommandToCommandBuffer(_ commandBuffer: MTLCommandBuffer, negativeSlope: Float, inputBuffer: MTLBuffer,
    metalDefaultLibrary:MTLLibrary, metalDevice:MTLDevice) -> MTLBuffer {
    
        print("==> addRectifierToCommandBuffer")
        
    let count = inputBuffer.length / MemoryLayout<Float>.size
    let (_, computePipelineState, _) = setupShaderInMetalPipeline("rectifier_linear", metalDefaultLibrary: metalDefaultLibrary,
        metalDevice: metalDevice)
    
    var negative = createFloatNumbersArray(1)
    negative[0] = negativeSlope
    
    let paramMetalBuffer = createFloatMetalBuffer(negative, metalDevice: metalDevice)
    
    // Create Metal Compute Command Encoder and add input and output buffers to it
    let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
    computeCommandEncoder.setBuffer(inputBuffer, offset: 0, at: 0)
    computeCommandEncoder.setBuffer(paramMetalBuffer, offset: 0, at: 1)
    // Set the shader function that Metal will use
    computeCommandEncoder.setComputePipelineState(computePipelineState!)
    
    // Set up thread groups on GPU
    // TODO: check out http://metalbyexample.com/introduction-to-compute/
    let threadsPerGroup = MTLSize(width:(computePipelineState?.threadExecutionWidth)!,height:1,depth:1)
    // ensure at least 1 threadgroup
    let numThreadgroups = MTLSize(width:(count-1)/(computePipelineState?.threadExecutionWidth)! + 1, height:1, depth:1)
    computeCommandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
    
    // Finalize configuration
    computeCommandEncoder.endEncoding()
        
        print(" <== addRectifierToCommandBuffer")
    
    return inputBuffer
}
