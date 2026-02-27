// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "LokmCore",
    platforms: [
        .iOS(.v16),
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "LokmCore",
            targets: ["LokmCore"]),
        .library(
            name: "LokmCamera",
            targets: ["LokmCamera"]),
        .library(
            name: "LokmVision",
            targets: ["LokmVision"]),
        .library(
            name: "LokmLLM",
            targets: ["LokmLLM"]),
    ],
    dependencies: [
        // 暂时零依赖，使用原生框架
    ],
    targets: [
        .target(
            name: "LokmCore",
            dependencies: []),
        .target(
            name: "LokmCamera",
            dependencies: ["LokmCore"]),
        .target(
            name: "LokmVision",
            dependencies: ["LokmCore"]),
        .target(
            name: "LokmLLM",
            dependencies: ["LokmCore"]),
        .testTarget(
            name: "LokmCoreTests",
            dependencies: ["LokmCore"]),
    ]
)