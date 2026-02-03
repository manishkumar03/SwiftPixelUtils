//
//  URLUtilities.swift
//  SwiftPixelUtils
//
//  Common URL utility functions
//

import Foundation

/// Internal URL utilities shared across the library.
enum URLUtilities {
    
    /// Check if a URL is a remote URL.
    /// Only `file://` scheme (or no scheme for relative paths) is considered local.
    /// All other schemes (http, https, ftp, sftp, smb, etc.) are considered remote.
    /// Remote URLs are not supported - users should download the data first.
    static func isRemoteURL(_ url: URL) -> Bool {
        guard let scheme = url.scheme?.lowercased() else {
            // No scheme means it's a relative path, which is local
            return false
        }
        // Only "file" scheme is local
        return scheme != "file"
    }
    
    /// Error message for when a remote URL is detected.
    static func remoteURLErrorMessage(example: String) -> String {
        "Remote URLs are not supported. Please download the image first and use .data(Data) instead. " +
        "Example: let data = try Data(contentsOf: url); \(example)"
    }
}
