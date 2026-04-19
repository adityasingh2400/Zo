// CameraPreview.swift — UIViewRepresentable wrapping AVCaptureVideoPreviewLayer.
//
// SwiftUI doesn't have a first-class preview layer abstraction. Wrap UIKit
// with a dedicated UIView subclass whose layer IS the preview layer, so
// AutoLayout / frame updates propagate without manually setting `frame` on
// every layout pass.

import AVFoundation
import SwiftUI
import UIKit

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    var gravity: AVLayerVideoGravity = .resizeAspectFill

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.backgroundColor = .black
        view.videoPreviewLayer.session = session
        view.videoPreviewLayer.videoGravity = gravity
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        // session + gravity only set once at make-time; SwiftUI updates
        // are no-ops unless we rebuild the layer.
    }

    /// UIView whose backing CALayer is a preview layer. Using a layer-
    /// backed view means `.frame` changes from SwiftUI auto-apply to the
    /// preview layer without manual observation.
    final class PreviewView: UIView {
        override static var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var videoPreviewLayer: AVCaptureVideoPreviewLayer {
            layer as! AVCaptureVideoPreviewLayer
        }
    }
}
