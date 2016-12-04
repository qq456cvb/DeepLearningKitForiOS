//
//  ViewController.swift
//  iOSDeepLearningKitApp
//
//  Created by Amund Tveit on 13/02/16.
//  Copyright © 2016 DeepLearningKit. All rights reserved.
//

import UIKit
import CoreGraphics
import AVFoundation

public class AtomicBoolean {
    private var val: UInt8 = 0
    
    public init(initialValue: Bool) {
        self.val = (initialValue == false ? 0 : 1)
    }
    
    public func getAndSet(value: Bool) -> Bool {
        if value {
            return  OSAtomicTestAndSet(7, &val)
        } else {
            return  OSAtomicTestAndClear(7, &val)
        }
    }
    
    public func get() -> Bool {
        return val != 0
    }
}

class ViewController: UIViewController, UIImagePickerControllerDelegate,
                                    UINavigationControllerDelegate, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    @IBOutlet weak var chooseBtn: UIButton!
    @IBOutlet weak var imageView: UIImageView!
    var deepNetwork: DeepNetwork!
    let imagePicker = UIImagePickerController()
    let path = Bundle.main.path(forResource: "yolo_tiny", ofType: "bson")!
    let imageShape:[Float] = [1.0, 3.0, 448.0, 448.0]
    let cameraSession = AVCaptureSession()
    let caching_mode = false
    var loaded = false
    var frame_done  = AtomicBoolean.init(initialValue: true)
    
    @IBAction func start(_ sender: Any) {
        
        cameraSession.sessionPreset = AVCaptureSessionPresetMedium
        
        let captureDevice = AVCaptureDevice.defaultDevice(withMediaType: AVMediaTypeVideo) as AVCaptureDevice
        do {
            let deviceInput = try AVCaptureDeviceInput(device: captureDevice)
            
            cameraSession.beginConfiguration() // 1
            
            if (cameraSession.canAddInput(deviceInput) == true) {
                cameraSession.addInput(deviceInput)
            }
            
            let dataOutput = AVCaptureVideoDataOutput() // 2
            
            dataOutput.videoSettings = [(kCVPixelBufferPixelFormatTypeKey as NSString) : NSNumber(value: kCVPixelFormatType_32BGRA as UInt32)] // 3
            
            dataOutput.alwaysDiscardsLateVideoFrames = true // 4
            
            if (cameraSession.canAddOutput(dataOutput) == true) {
                cameraSession.addOutput(dataOutput)
            }
            
            cameraSession.commitConfiguration() //5
            
            let queue = DispatchQueue(label: "video") // 6
            dataOutput.setSampleBufferDelegate(self, queue: queue) // 7
            
            cameraSession.startRunning()
            
        }
        catch let error as NSError {
            NSLog("\(error), \(error.localizedDescription)")
        }
    }
    
    func imageFromSampleBuffer(sampleBuffer : CMSampleBuffer) -> UIImage
    {
        // Get a CMSampleBuffer's Core Video image buffer for the media data
        let  imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
        // Lock the base address of the pixel buffer
        CVPixelBufferLockBaseAddress(imageBuffer!, CVPixelBufferLockFlags.readOnly);
        
        
        // Get the number of bytes per row for the pixel buffer
        let baseAddress = CVPixelBufferGetBaseAddress(imageBuffer!);
        
        // Get the number of bytes per row for the pixel buffer
        let bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer!);
        // Get the pixel buffer width and height
        let width = CVPixelBufferGetWidth(imageBuffer!);
        let height = CVPixelBufferGetHeight(imageBuffer!);
        
        // Create a device-dependent RGB color space
        let colorSpace = CGColorSpaceCreateDeviceRGB();
        
        // Create a bitmap graphics context with the sample buffer data
        var bitmapInfo: UInt32 = CGBitmapInfo.byteOrder32Little.rawValue
        bitmapInfo |= CGImageAlphaInfo.premultipliedFirst.rawValue & CGBitmapInfo.alphaInfoMask.rawValue
        //let bitmapInfo: UInt32 = CGBitmapInfo.alphaInfoMask.rawValue
        let context = CGContext.init(data: baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: bitmapInfo)
        // Create a Quartz image from the pixel data in the bitmap graphics context
        let quartzImage = context?.makeImage();
        // Unlock the pixel buffer
        CVPixelBufferUnlockBaseAddress(imageBuffer!, CVPixelBufferLockFlags.readOnly);
        
        // Create an image object from the Quartz image
        let image = UIImage.init(cgImage: quartzImage!, scale: 1.0, orientation: .down);
        
        return (image);
    }
    
    func captureOutput(_ captureOutput: AVCaptureOutput!, didOutputSampleBuffer sampleBuffer: CMSampleBuffer!, from connection: AVCaptureConnection!) {
        if frame_done.get() {
            _ = self.frame_done.getAndSet(value: false)
            let captured = self.imageFromSampleBuffer(sampleBuffer: sampleBuffer)
            DispatchQueue.global().async {
                let resized = self.resizeImage(image: captured, newWidth: CGFloat(self.imageShape[3]), newHeight: CGFloat(self.imageShape[2]))
                
                let (r, g, b, _) = imageToMatrix(resized)
                var image = b + g + r
                for (i, _) in image.enumerated() {
                    image[i] /= 255
                }
                self.deepNetwork.loadDeepNetworkFromBSON(self.path, inputImage: image, inputShape: self.imageShape, caching_mode:self.caching_mode)
                
                // 1. classify image (of cat)
                self.deepNetwork.yoloDetect(captured, imageView: self.imageView)
                _ = self.frame_done.getAndSet(value: true)
            }
        }
    }
    
    func resizeImage(image: UIImage, newWidth: CGFloat, newHeight: CGFloat) -> UIImage {
        UIGraphicsBeginImageContext(CGSize(width: newWidth, height: newHeight))
        image.draw(in: CGRect(x: 0, y: 0, width: newWidth, height: newHeight))
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        return newImage!
    }
    
    @IBAction func imageChoosen(_ sender: Any) {
        imagePicker.allowsEditing = false
        imagePicker.sourceType = .photoLibrary
        
        present(imagePicker, animated: true, completion: nil)
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any])  {
        print(info.debugDescription)
        if let pickedImage = info[UIImagePickerControllerOriginalImage] as? UIImage {
//            imageView.contentMode = .scaleAspectFit
            imageView.image = pickedImage
        }
        
//        DispatchQueue.global().async {
//            let resized = self.resizeImage(image: self.imageView.image!, newWidth: CGFloat(self.imageShape[3]), newHeight: CGFloat(self.imageShape[2]))
//            
//            //        print(resized.size.width)
//            
//            let (r, g, b, _) = imageToMatrix(resized)
//            var image = b + g + r
//            for (i, _) in image.enumerated() {
//                image[i] /= 255
//            }
//            self.deepNetwork.loadDeepNetworkFromBSON(self.path, inputImage: image, inputShape: self.imageShape, caching_mode:self.caching_mode)
//            
//            // 1. classify image (of cat)
//            self.deepNetwork.yoloDetect(image, imageView: self.imageView)
//        }
        
        dismiss(animated: true, completion: nil)
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        imagePicker.delegate = self
        imageView.image = #imageLiteral(resourceName: "lena")
    
        deepNetwork = DeepNetwork()
    }
    
    override func viewDidAppear(_ animated: Bool) {
        
       
        
        // conv1.json contains a cifar 10 image of a cat
//        let conv1Layer = deepNetwork.loadJSONFile("conv1")!
//        let image: [Float] = conv1Layer["input"] as! [Float]
//        
//        _ = UIImage(named: "lena")
//         //shows a tiny (32x32) CIFAR 10 image on screen
//        showCIFARImage(image)
        
        
//        var randomimage = createFloatNumbersArray(image.count)
//        for i in 0..<randomimage.count {
//            randomimage[i] = Float(arc4random_uniform(1000))
//        }
        
        
// **********************comment out below to debug at launch time ******************//
//        let imageCount = Int(imageShape.reduce(1, *))
//        
//        let resizeLena = resizeImage(image: #imageLiteral(resourceName: "lena"), newWidth: 448.0, newHeight: 448.0)
//        let (r, g, b, _) = imageToMatrix(resizeLena)
//        var image = b + g + r
//        for (i, _) in image.enumerated() {
//            image[i] /= 255
//        }
//        print(image.max()!)
//
//        var randomimage = createFloatNumbersArray(imageCount)
//        for i in 0..<randomimage.count {
//            randomimage[i] = Float(1.0)
//        }
//        
//        // 0. load network in network model
//        deepNetwork.loadDeepNetworkFromBSON(path, inputImage: image, inputShape: imageShape, caching_mode:caching_mode)
//        
//        // 1. classify image (of cat)
//        deepNetwork.yoloDetect(image, imageView: imageView)
        
//        deepNetwork.loadDeepNetworkFromBSON(path, inputImage: randomimage, inputShape: imageShape, caching_mode:caching_mode)
//        deepNetwork.classify(randomimage)
// **********************comment out above to debug at launch time ******************//
        
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    //***********************************************************************************
    
    func showCIFARImage(_ cifarImageData:[Float]) {
        var cifarImageData = cifarImageData
        let size = CGSize(width: 32, height: 32)
        let rect = CGRect(origin: CGPoint(x: 0,y: 0), size: size)
        
        UIGraphicsBeginImageContextWithOptions(size, false, 0)
        UIColor.white.setFill() // or custom color
        UIRectFill(rect)
        var image = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        // CIFAR 10 images are 32x32 in 3 channels - RGB
        // it is stored as 3 sequences of 32x32 = 1024 numbers in cifarImageData, i.e.
        // red: numbers from position 0 to 1024 (not inclusive)
        // green: numbers from position 1024 to 2048 (not inclusive)
        // blue: numbers from position 2048 to 3072 (not inclusive)
        for i in 0..<32 {
            for j in 0..<32 {
                let r = UInt8(cifarImageData[i*32 + j])
                let g = UInt8(cifarImageData[32*32 + i*32 + j])
                let b = UInt8(cifarImageData[2*32*32 + i*32 + j])
                
                // used to set pixels - RGBA into an UIImage
                // for more info about RGBA check out https://en.wikipedia.org/wiki/RGBA_color_space
                image = image?.setPixelColorAtPoint(CGPoint(x: j,y: i), color: UIImage.RawColorType(r,g,b,255))!
                
                // used to read pixels - RGBA from an UIImage
                _ = image?.getPixelColorAtLocation(CGPoint(x:i, y:j))
            }
        }
        print(image?.size ?? CGSize(width: 0, height: 0))
        
        // Displaying original image.
        let originalImageView:UIImageView = UIImageView(frame: CGRect(x: 20, y: 20, width: image!.size.width, height: image!.size.height))
        originalImageView.image = image
        self.view.addSubview(originalImageView)
    }
    
    
    
}
