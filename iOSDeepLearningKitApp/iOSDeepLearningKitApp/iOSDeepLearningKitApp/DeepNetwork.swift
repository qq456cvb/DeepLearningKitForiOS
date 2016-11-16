//
//  DeepNetwork.swift
//  MemkiteMetal
//
//  Created by Amund Tveit on 24/11/15.
//  Copyright © 2015 memkite. All rights reserved.
//

import Foundation
import Metal

open class DeepNetwork {
    var gpuCommandLayers: [MTLCommandBuffer] = []
    var namedDataLayers: [(String, MTLBuffer)] = []
    var imageBuffer: MTLBuffer!
    var metalDevice: MTLDevice!
    var metalDefaultLibrary: MTLLibrary!
    var metalCommandQueue: MTLCommandQueue!
    var deepNetworkAsDict: Dictionary<String, Any>! // for debugging perhaps
    var layer_data_caches: [Dictionary<String,MTLBuffer>] = []
    var pool_type_caches: [Dictionary<String,String>] = []
    var dummy_image: [Float]!
    var previous_shape: [Float]!
    var blob_cache: [Dictionary<String,([Float],[Float])>] = []
    
    public init() {
        setupMetal()
        deepNetworkAsDict = nil
    }
    
    open func loadDeepNetworkFromBSON(_ networkName: String, inputImage: [Float], inputShape:[Float], caching_mode:Bool) {
        print(" ==> loadDeepNetworkFromBSON(networkName=\(networkName)")
        if deepNetworkAsDict == nil {
            print("loading json file!")
//            let dic = readBson(file: networkName)
            deepNetworkAsDict = readBson(file: networkName)
        }
        
        
        // IMAGE INPUT HANDLING START - TODO: hardcode input dimensions,
        // and have random data, and then later overwrite the first buffer
        //let imageLayer = loadJSONFile("conv1")!
        //let imageData: [Float] = imageLayer["input"] as! [Float]
        print(inputImage.count)
        let imageTensor = createMetalBuffer(inputImage, metalDevice: metalDevice)
        
        //         preLoadMetalShaders(metalDevice, metalDefaultLibrary:metalDefaultLibrary)
        
        setupNetworkFromDict(deepNetworkAsDict, inputimage: imageTensor, inputshape: inputShape, caching_mode:caching_mode )
    }

    
    open func loadDeepNetworkFromJSON(_ networkName: String, inputImage: [Float], inputShape:[Float], caching_mode:Bool) {
        print(" ==> loadDeepNetworkFromJSON(networkName=\(networkName)")
        if deepNetworkAsDict == nil {
            print("loading json file!")
            deepNetworkAsDict = loadJSONFile(networkName)! as! Dictionary<String, Any>
        }
        
        
        // IMAGE INPUT HANDLING START - TODO: hardcode input dimensions,
        // and have random data, and then later overwrite the first buffer
        //let imageLayer = loadJSONFile("conv1")!
        //let imageData: [Float] = imageLayer["input"] as! [Float]
        print(inputImage.count)
        let imageTensor = createMetalBuffer(inputImage, metalDevice: metalDevice)

        //         preLoadMetalShaders(metalDevice, metalDefaultLibrary:metalDefaultLibrary)

        setupNetworkFromDict(deepNetworkAsDict, inputimage: imageTensor, inputshape: inputShape, caching_mode:caching_mode )
    }
    
    
    func setupMetal() {
        // Get access to iPhone or iPad GPU
        metalDevice = MTLCreateSystemDefaultDevice()
        
        // Queue to handle an ordered list of command buffers
        metalCommandQueue = metalDevice.makeCommandQueue()
        print("metalCommandQueue = \(Unmanaged.passUnretained(metalCommandQueue).toOpaque())")
        
        // Access to Metal functions that are stored in Shaders.metal file, e.g. sigmoid()
        metalDefaultLibrary = metalDevice.newDefaultLibrary()
    }
}
