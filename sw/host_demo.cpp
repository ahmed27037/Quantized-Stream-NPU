#include <array>
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
        // Deterministic but non-trivial pattern.
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

} // namespace

namespace {

bool starts_with(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() &&
           value.compare(0, prefix.size(), prefix) == 0;
}

} // namespace

int main(int argc, char** argv) {
    bool randomize = false;
    int extra_bits = 4;
    std::uint32_t seed = 0xC0FFEEu;

    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        if (arg == "--random") {
            randomize = true;
        } else if (starts_with(arg, "--seed=")) {
            seed = static_cast<std::uint32_t>(std::stoul(std::string(arg.substr(7))));
        } else if (starts_with(arg, "--extra-bits=")) {
            extra_bits = std::stoi(std::string(arg.substr(13)));
        } else {
            std::cerr << "Usage: " << argv[0]
                      << " [--random] [--seed=<value>] [--extra-bits=<value>]\n";
            return EXIT_FAILURE;
        }
    }

    std::mt19937 rng(seed);
    Matrix a = make_sample_matrix(randomize, rng);
    Matrix b = make_sample_matrix(randomize, rng);

    const Matrix raw = gemm(a, b);
    const Matrix relu_out = relu(raw);
    const auto streamed = stream_order(relu_out);

    const int required_width = min_acc_width(kDataWidth, kArraySize);
    const int configured_width = required_width + extra_bits;

    std::cout << "==== Host-Side Reference ====\n";
    std::cout << "ARRAY_SIZE: " << kArraySize << "  DATA_WIDTH: " << kDataWidth
              << "  REQUIRED_ACC_WIDTH: " << required_width
              << "  CONFIGURED_ACC_WIDTH: " << configured_width << "\n\n";

    print_matrix("Matrix A:", a);
    print_matrix("Matrix B:", b);
    print_matrix("Raw product (no activation):", raw);
    print_matrix("After ReLU activation:", relu_out);

    std::cout << "Streaming order (row-major, ReLU applied):\n";
    for (std::size_t idx = 0; idx < streamed.size(); ++idx) {
        std::cout << "  [" << std::setw(2) << idx << "] => " << std::setw(6) << streamed[idx] << '\n';
    }
    std::cout << "\n";

    std::cout << "Feed values into hardware column-by-column.\n"
                 "For column k, drive A(:,k) and B(k,:) on a_stream/b_stream respectively.\n";

    return EXIT_SUCCESS;
}
