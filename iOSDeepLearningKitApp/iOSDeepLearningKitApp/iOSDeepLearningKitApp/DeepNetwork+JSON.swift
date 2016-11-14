//
//  DeepNetwork+JSON.swift
//  MemkiteMetal
//
//  Created by Amund Tveit on 10/12/15.
//  Copyright © 2015 memkite. All rights reserved.
//

import Foundation

public extension DeepNetwork {
    
func loadJSONFile(_ filename: String) -> NSDictionary? {
    print(" ==> loadJSONFile(filename=\(filename)")
    
    do {
        let bundle = Bundle.main
        let path = bundle.path(forResource: filename, ofType: "json")!
        let jsonData = try? Data(contentsOf: URL(fileURLWithPath: path))
        print(" <== loadJSONFile")
        return try JSONSerialization.jsonObject(with: jsonData!, options: .allowFragments) as? NSDictionary
    } catch _ {
        return nil
    }
}
}
