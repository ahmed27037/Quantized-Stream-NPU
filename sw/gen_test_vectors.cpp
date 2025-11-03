#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

namespace {

constexpr int kArraySize  = 4;
constexpr int kDataWidth  = 8;
constexpr int kInt8Max    = (1 << (kDataWidth - 1)) - 1;
constexpr int kInt8Min    = -(1 << (kDataWidth - 1));

using Matrix = std::array<std::array<int, kArraySize>, kArraySize>;
using FMatrix = std::array<std::array<float, kArraySize>, kArraySize>;

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

QuantParams compute_quant_params(float min_val, float max_val) {
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

FMatrix make_float_matrix(bool randomize, std::mt19937& rng, int offset) {
    FMatrix m{};
    if (!randomize) {
        for (int r = 0; r < kArraySize; ++r) {
            for (int c = 0; c < kArraySize; ++c) {
                m[r][c] = (r == c) ? (r + 1.5f + offset) : 
                         ((r < c) ? (c - r + 0.5f) : -(r - c + 0.25f));
            }
        }
        return m;
    }

    std::uniform_real_distribution<float> dist(-30.0f, 30.0f);
    for (auto& row : m) {
        for (float& value : row) {
            value = dist(rng);
        }
    }
    return m;
}

} // namespace

int main(int argc, char** argv) {
    bool randomize = false;
    std::uint32_t seed = 0xDEADBEEFu;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--random") {
            randomize = true;
        } else if (arg.substr(0, 7) == "--seed=") {
            seed = static_cast<std::uint32_t>(std::stoul(arg.substr(7)));
        }
    }

    std::mt19937 rng(seed);
    
    FMatrix a_float = make_float_matrix(randomize, rng, 0);
    FMatrix b_float = make_float_matrix(randomize, rng, 1);

    float a_min, a_max, b_min, b_max;
    find_float_matrix_range(a_float, a_min, a_max);
    find_float_matrix_range(b_float, b_min, b_max);

    QuantParams a_params = compute_quant_params(a_min, a_max);
    QuantParams b_params = compute_quant_params(b_min, b_max);

    Matrix a = quantize_matrix(a_float, a_params);
    Matrix b = quantize_matrix(b_float, b_params);
    Matrix golden = gemm(a, b);

    // Write test vectors to file for SystemVerilog testbench
    std::ofstream vecfile("build/test_vectors.hex");
    if (!vecfile) {
        std::cerr << "ERROR: Could not open build/test_vectors.hex for writing\n";
        return EXIT_FAILURE;
    }

    // Write matrices
    for (int r = 0; r < kArraySize; ++r) {
        for (int c = 0; c < kArraySize; ++c) {
            vecfile << std::hex << (a[r][c] & 0xFF) << "\n";
        }
    }
    for (int r = 0; r < kArraySize; ++r) {
        for (int c = 0; c < kArraySize; ++c) {
            vecfile << std::hex << (b[r][c] & 0xFF) << "\n";
        }
    }
    for (int r = 0; r < kArraySize; ++r) {
        for (int c = 0; c < kArraySize; ++c) {
            vecfile << std::hex << (golden[r][c] & 0xFFFFFFFF) << "\n";
        }
    }
    vecfile.close();

    // Print human-readable summary
    std::cout << "========================================\n";
    std::cout << "  Quantized Test Vector Generator\n";
    std::cout << "========================================\n\n";
    
    std::cout << "Quantization Parameters:\n";
    std::cout << "  Matrix A: scale=" << a_params.scale << ", zero_point=" << a_params.zero_point << "\n";
    std::cout << "  Matrix B: scale=" << b_params.scale << ", zero_point=" << b_params.zero_point << "\n\n";

    std::cout << "Quantized Matrix A (INT8):\n";
    for (const auto& row : a) {
        for (int val : row) {
            std::cout << std::setw(5) << val << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    std::cout << "Quantized Matrix B (INT8):\n";
    for (const auto& row : b) {
        for (int val : row) {
            std::cout << std::setw(5) << val << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    std::cout << "Expected Golden Output (INT32):\n";
    for (const auto& row : golden) {
        for (int val : row) {
            std::cout << std::setw(8) << val << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    std::cout << "✓ Test vectors written to build/test_vectors.hex\n";
    std::cout << "✓ Ready for RTL simulation\n";

    return EXIT_SUCCESS;
}

