//
//  ViewController.swift
//  iOSDeepLearningKitApp
//
//  Created by Amund Tveit on 13/02/16.
//  Copyright © 2016 DeepLearningKit. All rights reserved.
//

import UIKit
import CoreGraphics

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    @IBOutlet weak var chooseBtn: UIButton!
    @IBOutlet weak var imageView: UIImageView!
    var deepNetwork: DeepNetwork!
    let imagePicker = UIImagePickerController()
    let path = Bundle.main.path(forResource: "yolo_tiny", ofType: "bson")!
    let imageShape:[Float] = [1.0, 3.0, 448.0, 448.0]
    let caching_mode = false
    var loaded = false
    
    func resizeImage(image: UIImage, newWidth: CGFloat, newHeight: CGFloat) -> UIImage {
        
//        let scale = newWidth / image.size.width
//        let newHeight = image.size.height * scale
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
        
        let resized = resizeImage(image: imageView.image!, newWidth: CGFloat(imageShape[3]), newHeight: CGFloat(imageShape[2]))
        
        print(resized.size.width)
        
        let (r, g, b, _) = imageToMatrix(resized)
        var image = b + g + r
        for (i, _) in image.enumerated() {
            image[i] /= 255
        }
        deepNetwork.loadDeepNetworkFromBSON(path, inputImage: image, inputShape: imageShape, caching_mode:caching_mode)

        
        // 1. classify image (of cat)
        deepNetwork.yoloDetect(image, imageView: imageView)
        
        dismiss(animated: true, completion: nil)
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        imagePicker.delegate = self
        imageView.image = #imageLiteral(resourceName: "lena")
        // 0. load network in network model
        
//        let resized = resizeImage(image: #imageLiteral(resourceName: "lena"), newWidth: CGFloat(imageShape[3]), newHeight: CGFloat(imageShape[2]))
//        
//        print(resized.size.width)
//        
//        let (r, g, b, _) = imageToMatrix(resized)
//        var image = b + g + r
//        for (i, _) in image.enumerated() {
//            image[i] /= 255
//        }
//        
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
        
        // 2. reset deep network and classify random image
//        deepNetwork.loadDeepNetworkFromJSON("nin_cifar10_full", inputImage: randomimage, inputShape: imageShape,caching_mode:caching_mode)
//        deepNetwork.classify(randomimage)
        
        // 3. reset deep network and classify cat image again
//        deepNetwork.loadDeepNetworkFromJSON("simple", inputImage: image, inputShape: imageShape,caching_mode:caching_mode)
//        deepNetwork.classify(image)
        
//        exit(0)
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
