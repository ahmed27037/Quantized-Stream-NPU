#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {

constexpr int kDataWidth  = 8;
constexpr int kInt8Max    = (1 << (kDataWidth - 1)) - 1;
constexpr int kInt8Min    = -(1 << (kDataWidth - 1));
constexpr float kTolerance = 1e-5f;

struct QuantParams {
    float scale;
    int zero_point;
    
    QuantParams(float s, int zp) : scale(s), zero_point(zp) {}
};

int8_t quantize(float value, const QuantParams& params) {
    int32_t quantized = static_cast<int32_t>(std::round(value / params.scale)) + params.zero_point;
    if (quantized < kInt8Min) return kInt8Min;
    if (quantized > kInt8Max) return kInt8Max;
    return static_cast<int8_t>(quantized);
}

float dequantize(int8_t value, const QuantParams& params) {
    return (static_cast<float>(value) - params.zero_point) * params.scale;
}

QuantParams compute_quant_params(float min_val, float max_val) {
    float abs_max = std::max(std::abs(min_val), std::abs(max_val));
    if (abs_max < 1e-8f) {
        return QuantParams(1.0f, 0);
    }
    float scale = abs_max / kInt8Max;
    return QuantParams(scale, 0);
}

// ============================================================================
// Test Cases
// ============================================================================

struct TestResult {
    std::string name;
    bool passed;
    std::string message;
};

TestResult test_symmetric_quantization() {
    float test_vals[] = {-127.0f, -50.0f, -1.0f, 0.0f, 1.0f, 50.0f, 127.0f};
    QuantParams params = compute_quant_params(-127.0f, 127.0f);
    
    if (std::abs(params.scale - 1.0f) > kTolerance) {
        return {"symmetric_quantization", false, "Expected scale=1.0 for [-127,127] range"};
    }
    if (params.zero_point != 0) {
        return {"symmetric_quantization", false, "Expected zero_point=0 for symmetric quantization"};
    }
    
    for (float val : test_vals) {
        int8_t q = quantize(val, params);
        float dq = dequantize(q, params);
        float error = std::abs(dq - val);
        if (error > 1.0f) {
            return {"symmetric_quantization", false, 
                    "Large reconstruction error: " + std::to_string(error)};
        }
    }
    
    return {"symmetric_quantization", true, ""};
}

TestResult test_range_clipping() {
    QuantParams params(1.0f, 0);
    
    // Test upper bound clipping
    int8_t q_max = quantize(200.0f, params);
    if (q_max != kInt8Max) {
        return {"range_clipping", false, 
                "Expected clipping to " + std::to_string(kInt8Max) + ", got " + std::to_string(q_max)};
    }
    
    // Test lower bound clipping
    int8_t q_min = quantize(-200.0f, params);
    if (q_min != kInt8Min) {
        return {"range_clipping", false,
                "Expected clipping to " + std::to_string(kInt8Min) + ", got " + std::to_string(q_min)};
    }
    
    return {"range_clipping", true, ""};
}

TestResult test_zero_preservation() {
    std::vector<std::pair<float, float>> ranges = {
        {-10.0f, 10.0f},
        {-127.0f, 127.0f},
        {-50.5f, 50.5f}
    };
    
    for (const auto& range : ranges) {
        QuantParams params = compute_quant_params(range.first, range.second);
        int8_t q_zero = quantize(0.0f, params);
        float dq_zero = dequantize(q_zero, params);
        
        if (std::abs(dq_zero) > kTolerance) {
            return {"zero_preservation", false,
                    "Zero not preserved: dequantized to " + std::to_string(dq_zero)};
        }
    }
    
    return {"zero_preservation", true, ""};
}

TestResult test_scaling() {
    float min_val = -50.0f;
    float max_val = 50.0f;
    QuantParams params = compute_quant_params(min_val, max_val);
    
    float expected_scale = 50.0f / kInt8Max;
    if (std::abs(params.scale - expected_scale) > kTolerance) {
        return {"scaling", false,
                "Expected scale=" + std::to_string(expected_scale) + 
                ", got " + std::to_string(params.scale)};
    }
    
    // Test that extremes map correctly
    int8_t q_max = quantize(50.0f, params);
    int8_t q_min = quantize(-50.0f, params);
    
    if (std::abs(q_max) < kInt8Max - 2 || std::abs(q_min) < kInt8Max - 2) {
        return {"scaling", false, "Extremes not utilizing full int8 range"};
    }
    
    return {"scaling", true, ""};
}

TestResult test_reconstruction_accuracy() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    
    QuantParams params = compute_quant_params(-100.0f, 100.0f);
    
    float max_error = 0.0f;
    constexpr int num_samples = 1000;
    
    for (int i = 0; i < num_samples; ++i) {
        float original = dist(rng);
        int8_t quantized = quantize(original, params);
        float reconstructed = dequantize(quantized, params);
        float error = std::abs(reconstructed - original);
        max_error = std::max(max_error, error);
    }
    
    // Error should be at most half the quantization step
    float max_acceptable_error = params.scale / 2.0f + kTolerance;
    if (max_error > max_acceptable_error) {
        return {"reconstruction_accuracy", false,
                "Max error " + std::to_string(max_error) + 
                " exceeds acceptable " + std::to_string(max_acceptable_error)};
    }
    
    return {"reconstruction_accuracy", true, ""};
}

TestResult test_edge_cases() {
    // Test with very small range
    QuantParams small_params = compute_quant_params(-0.001f, 0.001f);
    if (small_params.scale < 1e-10f) {
        return {"edge_cases", false, "Scale too small for tiny range"};
    }
    
    // Test with zero range
    QuantParams zero_params = compute_quant_params(0.0f, 0.0f);
    int8_t q = quantize(0.0f, zero_params);
    if (q != 0) {
        return {"edge_cases", false, "Zero range should produce zero output"};
    }
    
    // Test with asymmetric range
    QuantParams asym_params = compute_quant_params(-10.0f, 100.0f);
    if (asym_params.zero_point != 0) {
        return {"edge_cases", false, "Should still use symmetric quantization"};
    }
    
    return {"edge_cases", true, ""};
}

TestResult test_consistency() {
    // Same value should always quantize the same way
    QuantParams params(0.5f, 0);
    float test_val = 42.42f;
    
    int8_t q1 = quantize(test_val, params);
    int8_t q2 = quantize(test_val, params);
    int8_t q3 = quantize(test_val, params);
    
    if (q1 != q2 || q2 != q3) {
        return {"consistency", false, "Quantization is non-deterministic"};
    }
    
    // Dequantization should also be consistent
    float dq1 = dequantize(q1, params);
    float dq2 = dequantize(q1, params);
    
    if (std::abs(dq1 - dq2) > kTolerance) {
        return {"consistency", false, "Dequantization is non-deterministic"};
    }
    
    return {"consistency", true, ""};
}

} // namespace

int main() {
    std::cout << "========================================\n";
    std::cout << "  Quantization Verification Testbench  \n";
    std::cout << "========================================\n\n";
    
    std::vector<TestResult> results;
    
    results.push_back(test_symmetric_quantization());
    results.push_back(test_range_clipping());
    results.push_back(test_zero_preservation());
    results.push_back(test_scaling());
    results.push_back(test_reconstruction_accuracy());
    results.push_back(test_edge_cases());
    results.push_back(test_consistency());
    
    int passed = 0;
    int failed = 0;
    
    for (const auto& result : results) {
        std::cout << "[" << (result.passed ? "PASS" : "FAIL") << "] " 
                  << result.name;
        if (!result.passed) {
            std::cout << " - " << result.message;
        }
        std::cout << "\n";
        
        if (result.passed) {
            ++passed;
        } else {
            ++failed;
        }
    }
    
    std::cout << "\n========================================\n";
    std::cout << "Results: " << passed << " passed, " << failed << " failed\n";
    std::cout << "========================================\n";
    
    return (failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

