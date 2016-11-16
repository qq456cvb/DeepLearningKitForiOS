//
//  BsonUtil.swift
//  iOSDeepLearningKitApp
//
//  Created by Neil on 15/11/2016.
//  Copyright © 2016 DeepLearningKit. All rights reserved.
//

import Foundation

func readArr(stream: inout InputStream) -> Any {
    let intBuf = UnsafeMutablePointer<UInt8>.allocate(capacity: 4)
    let byteBuf = UnsafeMutablePointer<UInt8>.allocate(capacity: 1)
    
    stream.read(byteBuf, maxLength: 1)
    stream.read(intBuf, maxLength: 4)
    let byte = byteBuf[0]
    let cnt = Int(intBuf.deinitialize().assumingMemoryBound(to: UInt32.self)[0])
    if byte == 0 { // value
        let arrBuf = UnsafeMutablePointer<UInt8>.allocate(capacity: cnt * 4)
        stream.read(arrBuf, maxLength: cnt * 4)
        let floatArrBuf = arrBuf.deinitialize().assumingMemoryBound(to: Float.self)
        var arr = Array(repeating: 0.0, count: cnt) as! [Float]
        for i in 0...(cnt-1) {
            arr[i] = floatArrBuf[i]
        }
        arrBuf.deinitialize().deallocate(bytes: cnt * 4, alignedTo: 1)
        intBuf.deallocate(capacity: 4)
        byteBuf.deallocate(capacity: 1)
        return arr
    } else if byte == 1 { // dictionary
        var arr = [Any].init()
        for _ in 0...(cnt-1) {
            arr.append(readDic(stream: &stream))
        }
        intBuf.deallocate(capacity: 4)
        byteBuf.deallocate(capacity: 1)
        return arr
    } else if byte == 3 { // string
        var arr = [String].init()
        for _ in 0...(cnt-1) {
            stream.read(intBuf, maxLength: 4)
            let strLen = Int(intBuf.deinitialize().assumingMemoryBound(to: UInt32.self)[0])
            
            // read string
            let strBuf = UnsafeMutablePointer<UInt8>.allocate(capacity: strLen)
            stream.read(strBuf, maxLength: strLen)
            let data = Data(bytes: strBuf, count: strLen)
            let valStr = String(data: data, encoding: String.Encoding.ascii)
            arr.append(valStr!)
            strBuf.deallocate(capacity: strLen)
            
        }
        intBuf.deallocate(capacity: 4)
        byteBuf.deallocate(capacity: 1)
        return arr
    } else { // never get here
        assert(false)
    }
    assert(false)
    return [Any].init()
}

func readDic(stream: inout InputStream) -> Dictionary<String, Any> {
    var dic = Dictionary<String, Any>.init()
    let intBuf = UnsafeMutablePointer<UInt8>.allocate(capacity: 4)
    let byteBuf = UnsafeMutablePointer<UInt8>.allocate(capacity: 1)
    
    stream.read(intBuf, maxLength: 4)
    let num = intBuf.deinitialize().assumingMemoryBound(to: UInt32.self)[0]
    for _ in 0...(num-1) {
        stream.read(intBuf, maxLength: 4)
        let strLen = Int(intBuf.deinitialize().assumingMemoryBound(to: UInt32.self)[0])
        let strBuf = UnsafeMutablePointer<UInt8>.allocate(capacity: strLen)
        stream.read(strBuf, maxLength: strLen)
        
        var data = Data(bytes: strBuf, count: strLen)
        let str = String(data: data, encoding: String.Encoding.ascii)
        
        stream.read(byteBuf, maxLength: 1)
        let byte = byteBuf[0]
        if byte == 0 { // value
            stream.read(intBuf, maxLength: 4)
            let f = intBuf.deinitialize().assumingMemoryBound(to: Float.self)[0]
            dic[str!] = f
        } else if byte == 1 { // dictionary
            dic[str!] = readDic(stream: &stream)
        } else if byte == 2 { // array
            dic[str!] = readArr(stream: &stream)
        } else if byte == 3 { // string
            // read string length
            stream.read(intBuf, maxLength: 4)
            let innerStrLen = Int(intBuf.deinitialize().assumingMemoryBound(to: UInt32.self)[0])
            
            // read string
            let buf = UnsafeMutablePointer<UInt8>.allocate(capacity: innerStrLen)
            stream.read(buf, maxLength: innerStrLen)
            data = Data(bytes: buf, count: innerStrLen)
            let valStr = String(data: data, encoding: String.Encoding.ascii)
            
            dic[str!] = valStr
            buf.deallocate(capacity: innerStrLen)
        } else {
            assert(false) // cannot get in
        }
        strBuf.deallocate(capacity: strLen)
    }
    intBuf.deallocate(capacity: 4)
    byteBuf.deallocate(capacity: 1)
    return dic
}



func readBson(file: String) -> Dictionary<String, Any> {
    var stream = InputStream(fileAtPath: file)
    stream?.open()
    let dic = readDic(stream: &stream!)
    stream?.close()
    return dic
}
