//
//  MetalErrorsEnum.swift
//  AppleMetalDemoApp
//
//  Created by Manish Kumar on 2025-06-27.
//

// Metal kernel errors
enum MetalErrors: Error {
    case unsupportedDevice
    case libraryCreationFailed
    case makeFunctionFailed
    case commandQueueCreationFailed
}
