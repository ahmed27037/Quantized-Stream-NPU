#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string_view>
#include <vector>

namespace {

constexpr int kArraySize  = 4;
constexpr int kDataWidth  = 8;
constexpr int kInt8Max    = (1 << (kDataWidth - 1)) - 1;
constexpr int kInt8Min    = -(1 << (kDataWidth - 1));

using Matrix = std::array<std::array<int, kArraySize>, kArraySize>;
using FMatrix = std::array<std::array<float, kArraySize>, kArraySize>;

// ============================================================================
// Quantization Support
// ============================================================================

struct QuantParams {
    float scale;
    int zero_point;
    
    QuantParams() : scale(1.0f), zero_point(0) {}
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
    // Symmetric quantization for signed int8
    float abs_max = std::max(std::abs(min_val), std::abs(max_val));
    if (abs_max < 1e-8f) {
        return QuantParams(1.0f, 0);
    }
    float scale = abs_max / kInt8Max;
    return QuantParams(scale, 0);
}

Matrix quantize_matrix(const FMatrix& m, const QuantParams& params) {
    Matrix out{};
    for (int r = 0; r < kArraySize; ++r) {
        for (int c = 0; c < kArraySize; ++c) {
            out[r][c] = quantize(m[r][c], params);
        }
    }
    return out;
}

FMatrix dequantize_matrix(const Matrix& m, const QuantParams& params) {
    FMatrix out{};
    for (int r = 0; r < kArraySize; ++r) {
        for (int c = 0; c < kArraySize; ++c) {
            out[r][c] = dequantize(static_cast<int8_t>(m[r][c]), params);
        }
    }
    return out;
}

void find_float_matrix_range(const FMatrix& m, float& min_val, float& max_val) {
    min_val = m[0][0];
    max_val = m[0][0];
    for (const auto& row : m) {
        for (float value : row) {
            if (value < min_val) min_val = value;
            if (value > max_val) max_val = value;
        }
    }
}

// ============================================================================
// Matrix Operations
// ============================================================================

int min_acc_width(int data_width, int array_size) {
    const int multiplicand_bits = 2 * data_width;
    int guard_bits = 0;
    int max_partial = array_size - 1;
    while (max_partial > 0) {
        guard_bits++;
        max_partial >>= 1;
    }
    return multiplicand_bits + guard_bits;
}

void print_matrix(std::string_view label, const Matrix& m) {
    std::cout << label << '\n';
    for (const auto& row : m) {
        for (const auto& value : row) {
            std::cout << std::setw(6) << value << ' ';
        }
        std::cout << '\n';
    }
    std::cout << std::endl;
}

void print_float_matrix(std::string_view label, const FMatrix& m) {
    std::cout << label << '\n';
    for (const auto& row : m) {
        for (float value : row) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) << value << ' ';
        }
        std::cout << '\n';
    }
    std::cout << std::endl;
}

Matrix gemm(const Matrix& a, const Matrix& b) {
    Matrix c{};
    for (int i = 0; i < kArraySize; ++i) {
        for (int j = 0; j < kArraySize; ++j) {
            int sum = 0;
            for (int k = 0; k < kArraySize; ++k) {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
    return c;
}

FMatrix gemm_float(const FMatrix& a, const FMatrix& b) {
    FMatrix c{};
    for (int i = 0; i < kArraySize; ++i) {
        for (int j = 0; j < kArraySize; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < kArraySize; ++k) {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
    return c;
}

Matrix relu(const Matrix& m) {
    Matrix out = m;
    for (auto& row : out) {
        for (auto& value : row) {
            if (value < 0) {
                value = 0;
            }
        }
    }
    return out;
}

FMatrix relu_float(const FMatrix& m) {
    FMatrix out = m;
    for (auto& row : out) {
        for (float& value : row) {
            if (value < 0.0f) {
                value = 0.0f;
            }
        }
    }
    return out;
}

std::vector<int> stream_order(const Matrix& m) {
    std::vector<int> values;
    values.reserve(kArraySize * kArraySize);
    for (int r = 0; r < kArraySize; ++r) {
        for (int c = 0; c < kArraySize; ++c) {
            values.push_back(m[r][c]);
        }
    }
    return values;
}

Matrix make_sample_matrix(bool randomize, std::mt19937& rng) {
    Matrix m{};
    if (!randomize) {
        for (int r = 0; r < kArraySize; ++r) {
            for (int c = 0; c < kArraySize; ++c) {
                m[r][c] = (r == c) ? (r + 1) : ((r < c) ? (c - r) : -(r - c));
            }
        }
        return m;
    }

    std::uniform_int_distribution<int> dist(kInt8Min / 4, kInt8Max / 4);
    for (auto& row : m) {
        for (auto& value : row) {
            value = dist(rng);
        }
    }
    return m;
}

FMatrix make_float_matrix(bool randomize, std::mt19937& rng) {
    FMatrix m{};
    if (!randomize) {
        for (int r = 0; r < kArraySize; ++r) {
            for (int c = 0; c < kArraySize; ++c) {
                m[r][c] = (r == c) ? (r + 1.5f) : ((r < c) ? (c - r + 0.5f) : -(r - c + 0.25f));
            }
        }
        return m;
    }

    std::uniform_real_distribution<float> dist(-50.0f, 50.0f);
    for (auto& row : m) {
        for (float& value : row) {
            value = dist(rng);
        }
    }
    return m;
}

bool starts_with(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() &&
           value.compare(0, prefix.size(), prefix) == 0;
}

} // namespace

int main(int argc, char** argv) {
    bool randomize = false;
    bool use_quantization = false;
    int extra_bits = 4;
    std::uint32_t seed = 0xC0FFEEu;

    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        if (arg == "--random") {
            randomize = true;
        } else if (arg == "--quantize") {
            use_quantization = true;
        } else if (starts_with(arg, "--seed=")) {
            seed = static_cast<std::uint32_t>(std::stoul(std::string(arg.substr(7))));
        } else if (starts_with(arg, "--extra-bits=")) {
            extra_bits = std::stoi(std::string(arg.substr(13)));
        } else {
            std::cerr << "Usage: " << argv[0]
                      << " [--random] [--quantize] [--seed=<value>] [--extra-bits=<value>]\n";
            return EXIT_FAILURE;
        }
    }

    std::mt19937 rng(seed);
    
    const int required_width = min_acc_width(kDataWidth, kArraySize);
    const int configured_width = required_width + extra_bits;

    std::cout << "==== NPU Reference Implementation ====\n";
    std::cout << "ARRAY_SIZE: " << kArraySize << "  DATA_WIDTH: " << kDataWidth
              << "  REQUIRED_ACC_WIDTH: " << required_width
              << "  CONFIGURED_ACC_WIDTH: " << configured_width << "\n";
    std::cout << "Quantization: " << (use_quantization ? "ENABLED" : "DISABLED") << "\n\n";

    if (use_quantization) {
        // Floating-point workflow with quantization
        FMatrix a_float = make_float_matrix(randomize, rng);
        FMatrix b_float = make_float_matrix(randomize, rng);

        float a_min, a_max, b_min, b_max;
        find_float_matrix_range(a_float, a_min, a_max);
        find_float_matrix_range(b_float, b_min, b_max);

        QuantParams a_params = compute_quant_params(a_min, a_max);
        QuantParams b_params = compute_quant_params(b_min, b_max);

        std::cout << "Quantization Parameters:\n";
        std::cout << "  Matrix A: scale=" << a_params.scale << ", zero_point=" << a_params.zero_point << "\n";
        std::cout << "  Matrix B: scale=" << b_params.scale << ", zero_point=" << b_params.zero_point << "\n\n";

        Matrix a = quantize_matrix(a_float, a_params);
        Matrix b = quantize_matrix(b_float, b_params);

        print_float_matrix("Original Float Matrix A:", a_float);
        print_matrix("Quantized INT8 Matrix A:", a);
        print_float_matrix("Original Float Matrix B:", b_float);
        print_matrix("Quantized INT8 Matrix B:", b);

        // Compute in quantized domain
        const Matrix raw = gemm(a, b);
        const Matrix relu_out = relu(raw);
        const auto streamed = stream_order(relu_out);

        print_matrix("Quantized Product (no activation):", raw);
        print_matrix("After ReLU activation:", relu_out);

        // Also compute floating-point reference
        const FMatrix float_product = gemm_float(a_float, b_float);
        const FMatrix float_relu = relu_float(float_product);
        
        print_float_matrix("Float Reference Product:", float_product);
        print_float_matrix("Float Reference ReLU:", float_relu);

        std::cout << "Streaming order (row-major, ReLU applied, quantized):\n";
        for (std::size_t idx = 0; idx < streamed.size(); ++idx) {
            std::cout << "  [" << std::setw(2) << idx << "] => " << std::setw(6) << streamed[idx] << '\n';
        }
        std::cout << "\n";

    } else {
        // Direct INT8 workflow (legacy mode)
        Matrix a = make_sample_matrix(randomize, rng);
        Matrix b = make_sample_matrix(randomize, rng);

        const Matrix raw = gemm(a, b);
        const Matrix relu_out = relu(raw);
        const auto streamed = stream_order(relu_out);

        print_matrix("Matrix A:", a);
        print_matrix("Matrix B:", b);
        print_matrix("Raw product (no activation):", raw);
        print_matrix("After ReLU activation:", relu_out);

        std::cout << "Streaming order (row-major, ReLU applied):\n";
        for (std::size_t idx = 0; idx < streamed.size(); ++idx) {
            std::cout << "  [" << std::setw(2) << idx << "] => " << std::setw(6) << streamed[idx] << '\n';
        }
        std::cout << "\n";
    }

    std::cout << "Feed values into hardware column-by-column.\n"
                 "For column k, drive A(:,k) and B(k,:) on a_stream/b_stream respectively.\n";

    return EXIT_SUCCESS;
}
