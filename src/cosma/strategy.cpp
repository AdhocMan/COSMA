#include <cosma/strategy.hpp>
#include <options.hpp>

namespace cosma {
// constructors
Strategy::Strategy() = default;
// copy constructor
// Strategy::Strategy(Strategy& other) = default;
// move constructor
Strategy::Strategy(Strategy &&other) = default;

// constructs the Strategy from the command line
Strategy::Strategy(const std::string &cmd_line) {
    initialize(cmd_line);
    n_steps = divisors.size();
    check_if_valid();
    compute_min_sizes();
}

// constructs the Strategy form the command line
Strategy::Strategy(int argc, char **argv) {
    std::string input;
    for (auto i = 1; i < argc; ++i) {
        input += std::string(argv[i]) + " ";
    }
    initialize(input);
    n_steps = divisors.size();
    check_if_valid();
    compute_min_sizes();
}

Strategy::Strategy(int mm,
                   int nn,
                   int kk,
                   size_t PP,
                   std::vector<int> &divs,
                   std::string &dims,
                   std::string &types,
                   long long mem_limit,
                   double b,
                   bool top,
                   bool overlap,
                   bool busy_waiting)
    : m(mm)
    , n(nn)
    , k(kk)
    , P(PP)
    , memory_limit(mem_limit)
    , beta(b)
    , divisors(divs)
    , split_dimension(dims)
    , step_type(types)
    , n_steps(divisors.size())
    , topology(top)
    , overlap_comm_and_comp(overlap)
    , use_busy_waiting(busy_waiting) {
    n_steps = divisors.size();
    check_if_valid();
    compute_min_sizes();
}

Strategy::Strategy(int mm,
                   int nn,
                   int kk,
                   size_t PP,
                   std::string steps,
                   long long mem_limit,
                   double b,
                   bool top,
                   bool overlap,
                   bool busy_waiting)
    : m(mm)
    , n(nn)
    , k(kk)
    , P(PP)
    , memory_limit(mem_limit)
    , beta(b)
    , topology(top)
    , overlap_comm_and_comp(overlap)
    , use_busy_waiting(busy_waiting) {
    bool steps_predefined = options::flag_exists("-s", "--steps", steps);
    if (steps_predefined) {
        auto steps_it = options::find_flag(
            "-s", "--steps", "Division steps have to be defined.", steps);
        process_steps(steps_it, steps);
    } else {
        square_strategy();
    }
    n_steps = divisors.size();
    check_if_valid();
    compute_min_sizes();
}

Strategy::Strategy(int mm,
                   int nn,
                   int kk,
                   size_t PP,
                   long long mem_limit,
                   double b,
                   bool top,
                   bool overlap,
                   bool busy_waiting)
    : m(mm)
    , n(nn)
    , k(kk)
    , P(PP)
    , memory_limit(mem_limit)
    , beta(b)
    , topology(top)
    , overlap_comm_and_comp(overlap)
    , use_busy_waiting(busy_waiting) {
    // default_strategy();
    // spartition_strategy();
    square_strategy();
    // compress_steps();
    n_steps = divisors.size();
    check_if_valid();
    compute_min_sizes();
}

void Strategy::initialize(const std::string &cmd_line) {
    auto m_it = options::find_flag(
        "-m", "--m_dimension", "Dimension m has to be defined.", cmd_line);
    auto n_it = options::find_flag(
        "-n", "--n_dimension", "Dimension n has to be defined.", cmd_line);
    auto k_it = options::find_flag(
        "-k", "--k_dimension", "Dimension k has to be defined.", cmd_line);
    auto P_it = options::find_flag("-P",
                                   "--processors",
                                   "Number of processors has to be defined.",
                                   cmd_line);
    auto M_it = options::find_flag(
        "-L",
        "--memory",
        "Memory limit: maximum number of elements each rank can own.",
        cmd_line,
        false);
    m = options::next_int(m_it, cmd_line);
    n = options::next_int(n_it, cmd_line);
    k = options::next_int(k_it, cmd_line);
    P = options::next_int(P_it, cmd_line);
    memory_limit = options::next_long_long(M_it, cmd_line);

    // if memory limit not given, assume we have infinity
    // (i.e. assume that each rank can store all 3 matrices)
    if (memory_limit < 0) {
        memory_limit = std::numeric_limits<long long>::max();
    }
    topology = options::flag_exists("-t", "--topology", cmd_line);

    overlap_comm_and_comp = options::flag_exists("-o", "--overlap", cmd_line);

    if (options::flag_exists("-w", "--busy_waiting", cmd_line)) {
        use_busy_waiting = true;
    } else if (options::flag_exists("-l", "--polling", cmd_line)) {
        use_busy_waiting = false;
    } else {
        use_busy_waiting = true;
    }

    bool beta_predefined = options::flag_exists("-b", "--beta", cmd_line);
    if (beta_predefined) {
        auto beta_it =
            options::find_flag("-b",
                               "--beta",
                               "Beta parameter of gemm should be predefined.",
                               cmd_line);
        beta = options::next_double(beta_it, cmd_line);
    } else {
        beta = false;
    }

    bool steps_predefined = options::flag_exists("-s", "--steps", cmd_line);

    if (steps_predefined) {
        auto steps_it = options::find_flag(
            "-s", "--steps", "Division steps have to be defined.", cmd_line);
        process_steps(steps_it, cmd_line);
    } else {
        bool square_str = options::flag_exists("-q", "--square", cmd_line);
        bool spartition_str =
            options::flag_exists("-p", "--spartition", cmd_line);
        bool default_str = options::flag_exists("-d", "--default", cmd_line);

        bool compress_strategy =
            options::flag_exists("-c", "--compress", cmd_line);

        // default_strategy();
        // spartition_strategy();
        if (square_str) {
            square_strategy();
        } else if (spartition_str) {
            spartition_strategy();
        } else if (default_str) {
            default_strategy();
        } else {
            square_strategy();
        }

        if (compress_strategy) {
            compress_steps();
        }
    }
}

void Strategy::process_steps(size_t start, const std::string &line) {
    // go to the end of the string or space
    auto end = line.find(start, ' ');
    if (end == std::string::npos) {
        end = line.length();
    }
    std::string substring = line.substr(start, end);
    std::istringstream stream(substring);
    std::string token;

    while (std::getline(stream, token, ',')) {
        process_token(token);
    }
}

long long
Strategy::initial_memory(long long m, long long n, long long k, int P) {
    return math_utils::divide_and_round_up(m * n, P) +
           math_utils::divide_and_round_up(k * n, P) +
           math_utils::divide_and_round_up(m * k, P);
}

void Strategy::default_strategy() {
    if (empty()) 
        return;
    std::vector<int> factors = math_utils::decompose(P);
    long long m = this->m;
    long long n = this->n;
    long long k = this->k;
    int P = this->P;

    long long needed_memory = initial_memory(m, n, k, P);

    if (memory_limit < needed_memory) {
        throw_exception(
            std::string("This multiplication requires the memory ") +
            "for at least " + std::to_string(needed_memory) +
            " units, but only " + std::to_string(memory_limit) +
            " units are allowed. Either increase the memory limit " +
            "or change the strategy by using more sequential " + "steps.");
    }

    std::cout << "Default strategy" << std::endl;

    for (int i = 0; i < factors.size(); ++i) {
        bool did_parallel = false;
        int accumulated_div = 1;
        int next_div = factors[i];

        // m largest => split it
        while (m / accumulated_div >= std::max(n, k) &&
               needed_memory +
                       math_utils::divide_and_round_up(k * n * next_div, P) <=
                   memory_limit) {
            accumulated_div = next_div;
            did_parallel = true;
            ++i;
            if (i >= factors.size())
                break;
            next_div *= factors[i];
        }

        if (did_parallel) {
            i--;
            step_type += "p";
            split_dimension += "m";
            divisors.push_back(accumulated_div);
            needed_memory +=
                math_utils::divide_and_round_up(k * n * accumulated_div, P);
            m /= accumulated_div;
            P /= accumulated_div;
            continue;
        }

        // n largest => split it
        while (n / accumulated_div >= std::max(m, k) &&
               needed_memory +
                       math_utils::divide_and_round_up(k * m * next_div, P) <=
                   memory_limit) {
            accumulated_div = next_div;
            did_parallel = true;
            ++i;
            if (i >= factors.size())
                break;
            next_div *= factors[i];
        }

        if (did_parallel) {
            i--;
            step_type += "p";
            split_dimension += "n";
            divisors.push_back(accumulated_div);
            needed_memory +=
                math_utils::divide_and_round_up(k * m * accumulated_div, P);
            n /= accumulated_div;
            P /= accumulated_div;
            continue;
        }

        // k largest => split it
        while (k / accumulated_div >= std::max(m, n) &&
               needed_memory +
                       math_utils::divide_and_round_up(n * m * next_div, P) <=
                   memory_limit) {
            accumulated_div = next_div;
            did_parallel = true;
            ++i;
            if (i >= factors.size())
                break;
            next_div *= factors[i];
        }

        if (did_parallel) {
            i--;
            step_type += "p";
            split_dimension += "k";
            divisors.push_back(accumulated_div);
            needed_memory +=
                math_utils::divide_and_round_up(m * n * accumulated_div, P);
            k /= accumulated_div;
            P /= accumulated_div;
            continue;
        }

        // if parallel steps were not possible
        // then perform a sequential step first
        if (!did_parallel) {
            // don't count this iteration
            i--;
            step_type += "s";
            int div = 2;
            divisors.push_back(div);

            // if m largest => split it
            if (m >= std::max(k, n)) {
                split_dimension += "m";
                m /= div;
                continue;
            }

            // if n largest => split it
            if (n >= std::max(m, k)) {
                split_dimension += "n";
                n /= div;
                continue;
            }

            // if k largest => split it
            if (k >= std::max(m, n)) {
                split_dimension += "k";
                k /= div;
                continue;
            }
        }
    }
}

bool Strategy::divide(std::vector<int> &div_factors,
                      int &dim_i,
                      long long &dim1,
                      long long &dim2,
                      long long &dim3,
                      int &P,
                      long long &needed_memory,
                      const std::string label) {

    int next_div = 1;
    int accumulated_div = 1;
    bool did_parallel = false;

    if (dim_i < div_factors.size()) {
        next_div = div_factors[dim_i];
    }

    bool largest = dim1 >= std::max(dim2, dim3);
    bool first_run = true;

    // std::cout << "m-split divide and round = " <<
    // math_utils::divide_and_round_up(k * n * next_div, P) << std::endl;
    // std::cout << "m / acc_div = " << m/accumulated_div << std::endl;
    // if m largest => split it
    while (dim_i < div_factors.size() && (largest || first_run) &&
           needed_memory +
                   math_utils::divide_and_round_up(dim2 * dim3 * next_div, P) <=
               memory_limit) {
        accumulated_div = next_div;
        did_parallel = true;
        dim_i++;
        // i++;
        if (dim_i >= div_factors.size())
            break;
        next_div *= div_factors[dim_i];

        first_run = false;
        largest = dim1 / accumulated_div >= std::max(dim2, dim3);
    }

    if (did_parallel) {
        // i--;
        needed_memory +=
            math_utils::divide_and_round_up(dim2 * dim3 * accumulated_div, P);
        split_dimension += label;
        step_type += "p";
        divisors.push_back(accumulated_div);
        dim1 /= accumulated_div;
        P /= accumulated_div;
        // std::cout << "divided parallel m" << std::endl;
    }

    return did_parallel;
}

void Strategy::square_strategy() {
    long long m = this->m;
    long long n = this->n;
    long long k = this->k;
    int P = this->P;
    split_dimension = "";
    step_type = "";
    divisors.clear();

    long long needed_memory = initial_memory(m, n, k, P);

    if (memory_limit < needed_memory) {
        throw_exception(
            std::string("This multiplication requires the memory ") +
            "for at least " + std::to_string(needed_memory) +
            " units, but only " + std::to_string(memory_limit) +
            " units are allowed. Either increase the memory limit " +
            "or change the strategy by using more sequential " + "steps.");
    }

    while (P > 1) {
        int divm, divn, divk;
        std::tie(divm, divn, divk) = math_utils::balanced_divisors(m, n, k, P);

        // find prime factors of divm, divn, divk
        std::vector<int> divm_factors = math_utils::decompose(divm);
        std::vector<int> divn_factors = math_utils::decompose(divn);
        std::vector<int> divk_factors = math_utils::decompose(divk);

        int mi, ni, ki;
        mi = ni = ki = 0;

        int total_divisors =
            divm_factors.size() + divn_factors.size() + divk_factors.size();

        // Iterate through all prime factors of divm, divn and divk and
        // divide each dimensions with corresponding prime factors as long
        // as that dimension is the largest one.
        // Instead of dividing immediately m/divm, n/divn and k/divk,
        // it's always better to divide the dimension with smaller factors first
        // that are large enough to make that dimension NOT be the largest one
        // after division
        while (mi + ni + ki < total_divisors) {
            int i = mi + ni + ki;
            // std::cout << "i = " << i << ", m = " << m << ", n = " << n << ",
            // k = " << k << std::endl; std::cout << "mi = " << mi << ", ni = "
            // << ni << ", ki = " << ki << ", P = " << P << std::endl; std::cout
            // << "needed mem = " << needed_memory << ", mem limit = " <<
            // memory_limit << std::endl;
            bool did_parallel = false;

            long long mm = mi >= divm_factors.size() ? 1 : m;
            long long nn = ni >= divn_factors.size() ? 1 : n;
            long long kk = ki >= divk_factors.size() ? 1 : k;

            if (mm >= std::max(nn, kk)) {
                did_parallel =
                    divide(divm_factors, mi, m, n, k, P, needed_memory, "m");
                if (did_parallel)
                    continue;
            }

            if (nn >= std::max(mm, kk)) {
                did_parallel =
                    divide(divn_factors, ni, n, m, k, P, needed_memory, "n");
                if (did_parallel)
                    continue;
            }

            if (kk >= std::max(mm, nn)) {
                did_parallel =
                    divide(divk_factors, ki, k, m, n, P, needed_memory, "k");
                if (did_parallel)
                    continue;
            }

            // if not enough memory for any of the proposed parallel steps
            // then perform a single sequential step and recompute again
            // best divm, divn, divk for the smaller problem
            if (!did_parallel) {
                // don't count this iteration
                // i--;
                step_type += "s";
                int div = 2;
                divisors.push_back(div);

                // if m largest => split it
                if (m >= std::max(k, n)) {
                    split_dimension += "m";
                    m /= div;
                    break;
                }

                // if n largest => split it
                if (n >= std::max(m, k)) {
                    split_dimension += "n";
                    n /= div;
                    break;
                }

                // if k largest => split it
                if (k >= std::max(m, n)) {
                    split_dimension += "k";
                    k /= div;
                    break;
                }
            }
        }
    }

    std::string step_type_shorter = "";
    std::string split_dimension_shorter = "";
    std::vector<int> divisors_shorter;

    for (int i = 0; i < divisors.size(); ++i) {
        if (step_type[i] == 'p') {
            int div = divisors[i];
            while (i + 1 < divisors.size() && step_type[i + 1] == 'p' &&
                   split_dimension[i + 1] == split_dimension[i]) {
                div *= divisors[i + 1];
                i++;
            }
            step_type_shorter += "p";
            split_dimension_shorter += split_dimension[i];
            divisors_shorter.push_back(div);
            continue;
        }

        int j = i;
        int divm = 1;
        int divn = 1;
        int divk = 1;

        while (step_type[j] == 's') {
            if (split_dimension[j] == 'm')
                divm *= divisors[j];
            else if (split_dimension[j] == 'n')
                divn *= divisors[j];
            else
                divk *= divisors[j];
            j++;
        }

        if (divm > 1) {
            split_dimension_shorter += "m";
            step_type_shorter += "s";
            divisors_shorter.push_back(divm);
        }
        if (divn > 1) {
            split_dimension_shorter += "n";
            step_type_shorter += "s";
            divisors_shorter.push_back(divn);
        }
        if (divk > 1) {
            split_dimension_shorter += "k";
            step_type_shorter += "s";
            divisors_shorter.push_back(divk);
        }

        i = j - 1;
    }

    split_dimension = split_dimension_shorter;
    step_type = step_type_shorter;
    divisors = divisors_shorter;
}

double communication_cost(long long M,
                          long long N,
                          long long K,
                          int gridM,
                          int gridN,
                          int gridK) {
    int sizeM = M / gridM;
    int sizeN = N / gridN;
    int sizeK = K / gridK;
    long long tile_cost = sizeM * sizeN + sizeM * sizeK + sizeN * sizeK;
    double total_cost = 1.0 * tile_cost * gridM * gridN * gridK;
    return total_cost;
}

void processor_grid(unsigned p,
                    int M,
                    int N,
                    int K,
                    int &gridM,
                    int &gridN,
                    int &gridK,
                    double maxCompLoss = 0.03,
                    double maxCommLoss = 0.2) {

    unsigned lostProcesses = p - gridM * gridN * gridK;

    // if we loose too many processes, try to find something better
    double p_ratio = 1.0 * lostProcesses / p;
    if (p_ratio > maxCompLoss) {
        double optCommCost = communication_cost(M, N, K, gridM, gridN, gridK);
        double curCommCost = std::numeric_limits<double>::max();
        unsigned gridMcurrent, gridNcurrent, gridKcurrent;
        for (unsigned i = 0; i < p * maxCompLoss; i++) {
            if (1.0 * curCommCost / optCommCost > 1 + maxCommLoss) {
                std::tie(gridMcurrent, gridNcurrent, gridKcurrent) =
                    math_utils::balanced_divisors(M, N, K, p - i);
                curCommCost = communication_cost(
                    M, N, K, gridMcurrent, gridNcurrent, gridKcurrent);
            } else {
                gridM = gridMcurrent;
                gridN = gridNcurrent;
                gridK = gridKcurrent;
                break;
            }
        }
    }
}

void Strategy::spartition_strategy() {
    double load = std::cbrt(k) * std::cbrt(m) * std::cbrt(n) / std::cbrt(P);
    int a = (int)std::min(std::sqrt(memory_limit), load);
    int b = (int)std::max(1.0 * m * n * k / (P * memory_limit), load);

    int n_tiles_m = (m - 1) / a + 1;
    int n_tiles_n = (n - 1) / a + 1;
    int n_tiles_k = (k - 1) / b + 1;

    int tile_size_m = (m - 1) / n_tiles_m + 1;
    int tile_size_n = (n - 1) / n_tiles_n + 1;
    int tile_size_k = (k - 1) / n_tiles_k + 1;

    while (n_tiles_m * n_tiles_n * n_tiles_k > P) {
        if (a < std::sqrt(memory_limit)) {
            // find which dimension requires least stretching
            int new_tile_size_m = n_tiles_m == 1
                                      ? std::numeric_limits<int>::max()
                                      : (m - 1) / (n_tiles_m - 1) + 1;
            int new_tile_size_n = n_tiles_n == 1
                                      ? std::numeric_limits<int>::max()
                                      : (n - 1) / (n_tiles_n - 1) + 1;
            int new_tile_size_k = n_tiles_k == 1
                                      ? std::numeric_limits<int>::max()
                                      : (k - 1) / (n_tiles_k - 1) + 1;
            if (new_tile_size_k <= std::min(new_tile_size_m, new_tile_size_n)) {
                n_tiles_k = (k - 1) / new_tile_size_k + 1;
            } else {
                if (new_tile_size_n < new_tile_size_m &&
                    new_tile_size_n * tile_size_m < memory_limit) {
                    n_tiles_n = (n - 1) / new_tile_size_n + 1;
                } else if (new_tile_size_m * tile_size_n < memory_limit) {
                    n_tiles_m = (m - 1) / new_tile_size_m + 1;
                } else {
                    n_tiles_k = (k - 1) / new_tile_size_k + 1;
                }
            }
            if (n_tiles_m * n_tiles_n * n_tiles_k <= P) {
                break;
            }
        } else {
            n_tiles_k = P / (n_tiles_m * n_tiles_n);
            if (n_tiles_m * n_tiles_n * n_tiles_k <= P) {
                break;
            }
        }
    }

    // physical num cores refinement
    processor_grid(P, m, n, k, n_tiles_m, n_tiles_n, n_tiles_k);

    tile_size_m = (m - 1) / n_tiles_m + 1;
    tile_size_n = (n - 1) / n_tiles_n + 1;
    tile_size_k = (k - 1) / n_tiles_k + 1;

    double localMemCapacity =
        1.0 * memory_limit / std::max(tile_size_m, tile_size_n);

    std::vector<int> divisionsM = math_utils::decompose(n_tiles_m);
    std::vector<int> divisionsN = math_utils::decompose(n_tiles_n);
    std::vector<int> divisionsK = math_utils::decompose(n_tiles_k);

    // Find the optimal sequential schedule
    // squareSide = std::floor(std::sqrt(memory_limit + 1.0) - 1);
    int cubicSide = std::floor(std::sqrt(memory_limit / 3.0));
    a = std::min(cubicSide, tile_size_m);
    b = std::min(cubicSide, tile_size_n);
    int c = std::min(cubicSide, tile_size_k);

    int local_m = a;
    int local_n = b;
    int local_k = c;

    // We only do sequential schedule in K dimension
    int div = (local_k - 1) / c + 1;

    if (div > 1) {
        divisors.push_back(div);
        step_type += "s";
        split_dimension += "k";
        tile_size_m = local_m;
        tile_size_n = local_n;
        tile_size_k = local_k;
    }

    // Add parallel schedule
    int max_num_divisors = std::max(
        divisionsM.size(), std::max(divisionsN.size(), divisionsK.size()));

    for (size_t i = 0; i < max_num_divisors; i++) {
        if (divisionsM.size() > i) {
            step_type += "p";
            split_dimension += "m";
            divisors.push_back(divisionsM[i]);
        }
        if (divisionsN.size() > i) {
            step_type += "p";
            split_dimension += "n";
            divisors.push_back(divisionsN[i]);
        }
        if (divisionsK.size() > i) {
            step_type += "p";
            split_dimension += "k";
            divisors.push_back(divisionsK[i]);
        }
    }

    P = n_tiles_m * n_tiles_n * n_tiles_k;
}

// token is a triplet e.g. pm3 (denoting parallel (m / 3) step)
void Strategy::process_token(const std::string &step_triplet) {
    if (step_triplet.length() < 3)
        return;
    step_type += step_triplet[0];
    split_dimension += step_triplet[1];
    divisors.push_back(options::next_int(2, step_triplet));
}

void Strategy::throw_exception(const std::string &message) {
    std::cout << "Splitting strategy not well defined.\n";
    // std::cout << *this << std::endl;
    throw std::runtime_error(message);
}

bool Strategy::split_m(size_t i) const { return split_dimension[i] == 'm'; }

bool Strategy::split_n(size_t i) const { return split_dimension[i] == 'n'; }

bool Strategy::split_k(size_t i) const { return split_dimension[i] == 'k'; }

bool Strategy::split_A(size_t i) const { return split_m(i) || split_k(i); }

bool Strategy::split_B(size_t i) const { return split_k(i) || split_n(i); }

bool Strategy::split_C(size_t i) const { return split_m(i) || split_n(i); }

bool Strategy::split(char label, size_t step) const {
    if (label == 'A')
        return split_A(step);
    else if (label == 'B')
        return split_B(step);
    else
        return split_C(step);
}

bool Strategy::sequential_step(size_t i) const { return step_type[i] == 's'; }

bool Strategy::parallel_step(size_t i) const { return step_type[i] == 'p'; }

int Strategy::divisor(size_t i) const { return divisors[i]; }

int Strategy::divisor_m(size_t i) const { return split_m(i) ? divisors[i] : 1; }

int Strategy::divisor_n(size_t i) const { return split_n(i) ? divisors[i] : 1; }

int Strategy::divisor_k(size_t i) const { return split_k(i) ? divisors[i] : 1; }

int Strategy::divisor_row(char matrix, size_t i) const {
    if (matrix == 'A')
        return divisor_m(i);
    if (matrix == 'B')
        return divisor_k(i);
    if (matrix == 'C')
        return divisor_m(i);
    return 1;
}

int Strategy::divisor_col(char matrix, size_t i) const {
    if (matrix == 'A')
        return divisor_k(i);
    if (matrix == 'B')
        return divisor_n(i);
    if (matrix == 'C')
        return divisor_n(i);
    return 1;
}

bool Strategy::final_step(size_t i) const { return i == n_steps; }

int Strategy::parallel_steps_before_gemm(char label) const {
    if (label == 'A')
        return n_parallel_steps_before_gemm_a;
    if (label == 'B')
        return n_parallel_steps_before_gemm_b;
    if (label == 'C')
        return n_parallel_steps_before_gemm_c;
    return -1;
}

long long Strategy::required_memory(Strategy &strategy) {
    long long m = strategy.m;
    long long n = strategy.n;
    long long k = strategy.k;
    long long P = strategy.P;

    long long initial_size = initial_memory(m, n, k, P);

    for (int step = 0; step < strategy.n_steps; ++step) {
        int div = strategy.divisor(step);

        if (strategy.parallel_step(step)) {
            if (strategy.split_m(step))
                initial_size += math_utils::divide_and_round_up(k * n * div, P);
            else if (strategy.split_n(step))
                initial_size += math_utils::divide_and_round_up(m * k * div, P);
            else
                initial_size += math_utils::divide_and_round_up(m * n * div, P);
            P /= div;
        }

        m /= strategy.divisor_m(step);
        n /= strategy.divisor_n(step);
        k /= strategy.divisor_k(step);
    }

    return initial_size;
}

// checks if the strategy is well-defined
void Strategy::check_if_valid() {
    if (empty() && P != 1) {
        throw_exception("Strategy empty but number of ranks P != 1");
    }
#ifdef DEBUG
    std::cout << "Checking if the following strategy is valid: " << std::endl;
    std::cout << *this << std::endl;
#endif
    int mi = m;
    int ni = n;
    int ki = k;
    int Pi = P;

    n_parallel_steps = 0;
    n_parallel_steps_before_gemm_a = 0;
    n_parallel_steps_before_gemm_b = 0;
    n_parallel_steps_before_gemm_c = 0;

    for (size_t i = 0; i < n_steps; ++i) {
        if (divisors[i] <= 1) {
            throw_exception(
                std::string("Divisors in each step must be larger than 1.") +
                "Divisor in step " + std::to_string(i) + " = " +
                std::to_string(divisors[i]) + ".");
        }

        if (split_dimension[i] != 'm' && split_dimension[i] != 'n' &&
            split_dimension[i] != 'k') {
            throw_exception("Split dimension in each step must be m, n or k");
        }

        if (step_type[i] != 'p' && step_type[i] != 's') {
            throw_exception("Step type should be either p or s.");
        }

        if (step_type[i] == 'p') {
            n_parallel_steps++;
            if (!split_A(i)) {
                n_parallel_steps_before_gemm_a++;
            }
            if (!split_B(i)) {
                n_parallel_steps_before_gemm_b++;
            }
            if (!split_C(i)) {
                n_parallel_steps_before_gemm_c++;
            }

            if (Pi <= 1) {
                throw_exception(
                    std::string(
                        "Not enough processors for this division strategy.") +
                    "The product of all divisors in a parallel step should be "
                    "equal " +
                    "to the number of processors");
            }

            if (Pi % divisors[i] != 0) {
                throw_exception(std::string("The number of processors left in "
                                            "each parallel step ") +
                                "should be divisible by divisor.");
            }

            Pi /= divisors[i];
        } else {
            n_sequential_steps++;
            if (split_A(i)) {
                n_parallel_steps_before_gemm_a = 0;
            }
            if (split_B(i)) {
                n_parallel_steps_before_gemm_b = 0;
            }
            if (split_C(i)) {
                n_parallel_steps_before_gemm_c = 0;
            }
        }

        if (split_dimension[i] == 'm') {
            mi /= divisors[i];
        }

        // if last step, check if #columns >= #processors that share this block
        // of matrix we only check dimensions n and k, because these are the
        // dimensions defining the number of columns, i.e. dimension m does not
        // denote the #columns of any matrix
        if (i == n_steps - 1) {
            // since we are using column major ordering, the #columns of each
            // matrix must be at least the number of processors left at that
            // step
            if (split_dimension[i] == 'n') {
                ni /= divisors[i];
                if (ni < Pi) {
                    throw_exception(std::string("Dimension n at step ") +
                                    std::to_string(i) + " = " +
                                    std::to_string(ni) +
                                    ", which is less than the number of "
                                    "processors left = " +
                                    std::to_string(Pi));
                }
            }

            if (split_dimension[i] == 'k') {
                ki /= divisors[i];
                if (ki < Pi) {
                    throw_exception(
                        std::string("Dimension k at step ") +
                        std::to_string(i) + " = " + std::to_string(ki) +
                        ", which is less than the number " +
                        "of processors left = " + std::to_string(Pi));
                }
            }
        }
    }
    if (Pi != 1) {
        throw_exception(
            std::string(
                "Too many processors. The number of processors should be ") +
            "equal to the product of divisors in all parallel steps.");
    }

    memory_used = required_memory(*this);
    // check if we have enough memory for this splitting strategy
    /*
    if (memory_limit < memory_used) {
        throw_exception(std::string("The splitting strategy requires memory for
    roughly ")
                + std::to_string(memory_used) + " elements, but the memory limit
    is only "
                + std::to_string(memory_limit) + " elements. Either increase the
    memory limit "
                + "or change the strategy. (Hint: you could use some sequential
    "
                + "steps to decrease the required memory.)");
    }
    */
}

void Strategy::compress_steps() {
    int p_divm = 1;
    int p_divn = 1;
    int p_divk = 1;
    int s_divm = 1;
    int s_divn = 1;
    int s_divk = 1;

    for (size_t i = 0; i < split_dimension.size(); ++i) {
        if (parallel_step(i)) {
            p_divm *= divisor_m(i);
            p_divn *= divisor_n(i);
            p_divk *= divisor_k(i);
        } else {
            s_divm *= divisor_m(i);
            s_divn *= divisor_n(i);
            s_divk *= divisor_k(i);
        }
    }

    std::vector<int> divs = {p_divm, p_divn, p_divk, s_divm, s_divn, s_divk};

    n_steps = 0;
    divisors = std::vector<int>();
    split_dimension = "";
    step_type = "";

    for (size_t i = 0; i < divs.size(); ++i) {
        if (divs[i] > 1) {
            n_steps++;
            divisors.push_back(divs[i]);

            if (i < 3) {
                step_type += "p";
            } else {
                step_type += "s";
            }

            if (i % 3 == 0) {
                split_dimension += "m";
            } else if (i % 3 == 1) {
                split_dimension += "n";
            } else {
                split_dimension += "k";
            }
        }
    }
}

void Strategy::compute_min_sizes() {
    min_m = m;
    min_n = n;
    min_k = k;
    for (int step = 0; step < n_steps; ++step) {
        min_m /= divisor_m(step);
        min_n /= divisor_n(step);
        min_k /= divisor_k(step);
    }
}

bool Strategy::should_overlap_comm_and_comp(int step) const {
    bool last_step = step == n_steps - 1;
    if (!last_step) {
        return false;
    }

    int div = divisor(step);
    int divm = divisor_m(step);
    int divn = divisor_n(step);
    int divk = divisor_k(step);

    int newm = min_m;
    int newn = min_n;
    int newk = min_k;

    // overlap requires that the number of columns of the expanded matrix
    // i.e. the matrix that is not split is >= div, so that it can be split as
    // well
    bool overlap_possible = (split_m(step) && min_n >= div) ||
                            (split_n(step) && min_k >= div) ||
                            (split_k(step) && min_n >= div);

    if (split_m(step)) {
        newn /= div;
    } else if (split_n(step)) {
        newk /= div;
    } else {
        newn /= div;
    }

    bool overlap_turned_on = overlap_comm_and_comp;
    double score_no_overlap = math_utils::square_score(min_m, min_n, min_k);
    double score_with_overlap = math_utils::square_score(newm, newn, newk);
    auto diff = score_with_overlap - score_no_overlap;
    bool should_overlap = diff / score_no_overlap >= 0.5;

    // std::cout << "overlap_possible = " << overlap_possible << std::endl;
    // std::cout << "last_step = " << last_step << std::endl;
    // std::cout << "overlap_turned_on = " << overlap_turned_on << std::endl;
    // std::cout << "score_no_overlap = " << score_no_overlap << std::endl;
    // std::cout << "score_with_overlap = " << score_with_overlap << std::endl;
    // std::cout << "should_overlap = " << should_overlap << std::endl;
    bool condition = overlap_possible && overlap_turned_on && should_overlap;

#ifdef DEBUG
    std::cout << "Overlapping communication and computation." << std::endl;
#endif

    // return condition;
    return condition;
}

bool Strategy::empty() const {
    return n_steps == 0;
}

int Strategy::n_rows(char label) const {
    if (label == 'A') 
        return m;
    if (label == 'B') 
        return k;
    if (label == 'C') 
        return m;
    return -1;
}

int Strategy::n_cols(char label) const {
    if (label == 'A') 
        return k;
    if (label == 'B') 
        return n;
    if (label == 'C') 
        return n;

    return -1;
}

std::ostream &operator<<(std::ostream &os, const Strategy &other) {
    os << "Matrix dimensions (m, n, k) = (" << other.m << ", " << other.n
       << ", " << other.k << ")\n";
    os << "Number of processors: " << other.P << "\n";
    if (other.topology) {
        os << "Communication-aware topology turned on.\n";
    }
    if (other.overlap_comm_and_comp) {
        os << "Overlap of communication and computation: ON.\n";
        if (other.use_busy_waiting) {
            os << "Communication-thread policy (for overlap): "
               << "busy-waiting (using blocking one-sided MPI).\n";
        } else {
            os << "Communication-thread policy (for overlap): "
               << "polling (using non-blocking one-sided MPI).\n";
        }
    } else {
        os << "Overlap of communication and computation: OFF.\n";
    }
    os << "Divisions strategy: \n";
    for (size_t i = 0; i < other.n_steps; ++i) {
        if (other.step_type[i] == 'p') {
            os << "parallel (" << other.split_dimension[i] << " / "
               << other.divisors[i] << ")\n";
        } else {
            os << "sequential (" << other.split_dimension[i] << " / "
               << other.divisors[i] << ")\n";
        }
    }
    os << "Required memory per rank (in #elements): " << other.memory_used
       << "\n";
    os << "Available memory per rank (in #elements): " << other.memory_limit
       << "\n";
    return os;
}
} // namespace cosma
