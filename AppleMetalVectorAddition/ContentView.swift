//
//  ContentView.swift
//  AppleMetalDemoApp
//
//  Created by Manish Kumar on 2025-06-27.
//

import SwiftUI

struct ContentView: View {
    @State private var result: [Float] = []
    @State private var kernelExecutionTimeString = "0.0 ms"

    var body: some View {
        VStack(spacing: 16) {
            Text("Vector Addition Using Metal")
                .font(.title2.bold())
                .foregroundColor(.primary)

            Button(action: {
                (self.kernelExecutionTimeString, self.result) = MetalKernelDispatcher().executeMetalKernel()
            }) {
                Label("Run Kernel", systemImage: "bolt.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .tint(.accentColor)

            VStack(alignment: .leading, spacing: 8) {
                Text("Result:")
                    .font(.caption)
                    .foregroundColor(.secondary)

                Text(result[0..<min(10, result.count)]
                    .map { String(format: "%.1f", $0) }
                    .joined(separator: ", ") + "...")
                .font(.system(.body, design: .monospaced))
                .foregroundColor(.primary)

                Spacer().frame(height: 16)

                Text("Kernel execution time:")
                    .font(.caption)
                    .foregroundColor(.secondary)

                Text(self.kernelExecutionTimeString)
                    .font(.system(.body, design: .monospaced))
                    .foregroundColor(.primary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding()
        .background(Color.orange.opacity(0.1))
        .cornerRadius(20)
        .shadow(color: .black.opacity(0.1), radius: 10, x: 0, y: 5)
        .padding(.horizontal, 24)
    }
}
