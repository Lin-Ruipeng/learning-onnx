# Use a CPU baseline and conditional hardware backend tracks

Every Core Course lesson must pass on ONNX Runtime CPU, while GPU, NPU, and device-specific work lives in Conditional Backend Labs with capability detection and Hardware Acceptance Reports. This keeps the required path reproducible in local development and CI without weakening the depth of RK3588, HarmonyOS, TensorRT, or OpenVINO experiments or falsely claiming that unavailable hardware was tested.
