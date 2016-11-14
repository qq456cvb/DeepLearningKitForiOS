//
//  FullyConnectedLayer.swift
//  iOSDeepLearningKitApp
//
//  Created by Neil on 12/11/2016.
//  Copyright © 2016 DeepLearningKit. All rights reserved.
//

import Foundation
import Metal

//func getDataFromBlob(_ blob: NSDictionary) -> ([Float], [Float]) {
//    
//    let shape = blob["shape"] as! NSDictionary
//    let data = blob["data"] as! [Float]
//    var FloatData = createFloatNumbersArray(data.count)
//    for i in 0 ..< data.count {
//        FloatData[i] = data[i]
//    }
//    return (shape["dim"] as! [Float], FloatData)
//}

func createFullyConnectedLayerCached(_ layer: NSDictionary,
                                  inputBuffer: MTLBuffer,
                                  inputShape: [Float],
                                  metalCommandQueue: MTLCommandQueue, metalDefaultLibrary:MTLLibrary, metalDevice:MTLDevice,
                                  layer_data_caches: inout [Dictionary<String,MTLBuffer>],
                                  blob_cache: inout [Dictionary<String,([Float],[Float])>],
                                  layer_number: Int,
                                  layer_string: String, caching_mode:Bool) -> (MTLBuffer, MTLCommandBuffer, [Float]) {
    
    _ = Date()
    
    
    //        let metalCommandBuffer = metalCommandQueue.commandBuffer()
    let metalCommandBuffer = metalCommandQueue.makeCommandBufferWithUnretainedReferences()
    
    var fullyconnected_params_dict:NSDictionary = NSDictionary()
    var blobs:[NSDictionary] = []
    var weights:[Float] = []
    var weight_shape:[Float] = []
    var bias_data:[Float] = []
    var result_shape:[Float] = []
    var outputCount:Int = 0
    
    var input_dimensions:MetalFCTensorDimensions = MetalFCTensorDimensions(rows: 0, cols: 0)
    var weight_dimensions:MetalFCTensorDimensions = MetalFCTensorDimensions(rows: 0, cols: 0)
    var result_dimensions:MetalFCTensorDimensions = MetalFCTensorDimensions(rows: 0, cols: 0)
    var tensor_dimensions:[MetalFCTensorDimensions] = []
    
    
    if(!caching_mode) {
        print("NOTCACHINGMODE")
        fullyconnected_params_dict = layer["inner_product_param"] as! NSDictionary
        var num_output: Float = 0.0
        if let val = fullyconnected_params_dict["num_output"] as? Float {
            num_output = val
        } else {
            print("-------------Warning: no output found in fc layers")
        }
        
        _ = Date()
        
        
        if let tmpval = blob_cache[layer_number]["0"] {
            (weight_shape, weights) = tmpval
        } else {
            blobs = layer["blobs"] as! [NSDictionary]
            (weight_shape, weights) = getDataFromBlob(blobs[0])
            blob_cache[layer_number]["0"] = (weight_shape, weights)
        }
        
        blobs = layer["blobs"] as! [NSDictionary]
        (_, bias_data) = getDataFromBlob(blobs[1])
        
        
        // Create input and output vectors, and corresponding metal buffer
        let n = inputShape[0]
        result_shape = [n, weight_shape[0]]
        input_dimensions = MetalFCTensorDimensions(rows: n, cols: weight_shape[1])
        weight_dimensions = MetalFCTensorDimensions(rows: weight_shape[0], cols: weight_shape[1])
        result_dimensions = MetalFCTensorDimensions(rows: n, cols: weight_shape[0])
        tensor_dimensions = [input_dimensions, weight_dimensions, result_dimensions]
        
        outputCount = Int(n) * Int(num_output)
    }
    
    
    let resultBuffer = addFullyConnectedCommandToCommandBufferCached(metalCommandBuffer, inputBuffer: inputBuffer, weights: weights, outputCount: outputCount, tensor_dimensions: tensor_dimensions, bias: bias_data, metalDefaultLibrary: metalDefaultLibrary, metalDevice: metalDevice, layer_data_caches: &layer_data_caches, layer_number: layer_number, layer_string: layer_string, caching_mode: caching_mode)
    //metalCommandBuffer.commit()
    
    
    
    return (resultBuffer, metalCommandBuffer, result_shape)
    
}

func addFullyConnectedCommandToCommandBufferCached(_ commandBuffer: MTLCommandBuffer,
                                                inputBuffer: MTLBuffer,
                                                weights: [Float],
                                                outputCount: Int,
                                                tensor_dimensions: [MetalFCTensorDimensions],
                                                bias: [Float],
                                                metalDefaultLibrary:MTLLibrary, metalDevice:MTLDevice,
                                                layer_data_caches: inout [Dictionary<String,MTLBuffer>],
                                                layer_number: Int,
                                                layer_string: String, caching_mode:Bool) -> MTLBuffer {
    
    _ = Date()
    
    print("before output")
    
    var output:[Float] = []
    
    if(!caching_mode) {
        output = createFloatNumbersArray(outputCount)
    }
    
    print("before setupshaderinpipeline")
    
    let resultMetalBuffer = createOrReuseFloatMetalBuffer("resultMetalBuffer", data: output, cache: &layer_data_caches, layer_number: layer_number, metalDevice: metalDevice)
    
    print("after resultmetalbuffer")
    
    let weightMetalBuffer = createOrReuseFloatMetalBuffer("weightMetalBuffer", data: weights, cache: &layer_data_caches, layer_number:layer_number, metalDevice: metalDevice)
    
    let tensorDimensionsMetalBuffer = createOrReuseFCTensorDimensionsVectorMetalBuffer("tensorDimensionsMetalBuffer", data: tensor_dimensions, cache: &layer_data_caches, layer_number: layer_number, metalDevice: metalDevice)
    
    let biasMetalBuffer = createOrReuseFloatMetalBuffer("bias", data: bias, cache: &layer_data_caches, layer_number:layer_number, metalDevice: metalDevice)
    
    let (_, fullyConnectedComputePipelineState, _) = setupShaderInMetalPipeline("fc_layer", metalDefaultLibrary: metalDefaultLibrary, metalDevice: metalDevice)
    let metalComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder()
    
    // Create Metal Compute Command Encoder and add input and output buffers to it
    metalComputeCommandEncoder.setBuffer(resultMetalBuffer, offset: 0, at: 0)
    metalComputeCommandEncoder.setBuffer(weightMetalBuffer, offset: 0, at: 1)
    metalComputeCommandEncoder.setBuffer(tensorDimensionsMetalBuffer, offset: 0, at: 2)
    metalComputeCommandEncoder.setBuffer(inputBuffer, offset: 0, at: 3)
    metalComputeCommandEncoder.setBuffer(biasMetalBuffer, offset: 0, at: 4)
    
    // Set the shader function that Metal will use
    metalComputeCommandEncoder.setComputePipelineState(fullyConnectedComputePipelineState!)
    
    // Set up thread groups on GPU
    // TODO: check out http://metalbyexample.com/introduction-to-compute/
    let threadsPerGroup = MTLSize(width:(fullyConnectedComputePipelineState?.threadExecutionWidth)!,height:1,depth:1)
    // ensure at least 1 threadgroup
    let numThreadgroups = MTLSize(width:(outputCount-1)/(fullyConnectedComputePipelineState?.threadExecutionWidth)! + 1, height:1, depth:1)
    metalComputeCommandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
    
    // Finalize configuration
    metalComputeCommandEncoder.endEncoding()
    
    
    
    
    return resultMetalBuffer
    
}

