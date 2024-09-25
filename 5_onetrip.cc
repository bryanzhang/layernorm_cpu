#include <iostream>
#include <random>
#include <chrono>
#include <emmintrin.h>
#include <immintrin.h>
#include <sys/resource.h>
#include <unistd.h>

using namespace std;

inline float sum_vfloat16(__m512 vec) {
    __m256 hi256 = _mm512_extractf32x8_ps(vec, 1);
    __m256 lo256 = _mm512_extractf32x8_ps(vec, 0);
    __m256 vb = _mm256_hadd_ps(hi256, lo256);
    __m256 vc = _mm256_hadd_ps(vb, vb);
    __m256 vd = _mm256_hadd_ps(vc, vc);
    __m128 lo = _mm256_extractf128_ps(vd, 0);
    __m128 hi = _mm256_extractf128_ps(vd, 1);
    return (_mm_cvtss_f32(lo) + _mm_cvtss_f32(hi));
}

// GPT-2 layernorm forward pass
template<int B, int T, int C>  // 分别是batch size、context length和embdding dim
void layernorm_forward_cpu(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias) {
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            const float* x = inp + b * T * C + t * C;
            // calculate the mean
	    // _mm_prefetch(x, _MM_HINT_T0);
	    __m512 v_sum = _mm512_setzero_ps();
	    __m512 v_sqr = _mm512_setzero_ps();
	    for (int i = 0; i < C; i += 16) {
		// _mm_prefetch(x + i + 1, _MM_HINT_T0);
		__m512 va = _mm512_loadu_ps(x + i);
		v_sum = _mm512_add_ps(v_sum, va);
		v_sqr = _mm512_fmadd_ps(va, va, v_sqr);
	    }
	    // float m = sum_vfloat16(va) / C;
	    float m = _mm512_reduce_add_ps(v_sum) / C;
	    float m_sqr = _mm512_reduce_add_ps(v_sqr) / C;
	    float s = 1.0f / sqrtf(m_sqr - m * m + 1e-5f);

            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
	    __m512 broadcast_rstd = _mm512_set1_ps(s);
	    __m512 broadcast_tosub = _mm512_set1_ps(m * s);
            for (int i = 0; i < C; i += 16) {
		__m512 va = _mm512_loadu_ps(x + i);
		__m512 vb = _mm512_fmsub_ps(va, broadcast_rstd, broadcast_tosub);
		__m512 vw = _mm512_loadu_ps(weight + i);
		__m512 vbias = _mm512_loadu_ps(bias + i);
		__m512 vv = _mm512_fmadd_ps(vb, vw, vbias);
		_mm512_storeu_ps(out_bt + i, vv);
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

int main() {
        // 设置进程的nice值，-20是最高优先级
        if (setpriority(PRIO_PROCESS, getpid(), -20) == -1) {
          perror("setpriority");
          return 1;
        }

    	cpu_set_t cpuset;
	// 初始化CPU集合
	CPU_ZERO(&cpuset);
	CPU_SET(0, &cpuset);
	if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) == -1) {
		perror("sched_setaffinity");
		exit(EXIT_FAILURE);
	}

      	std::cout << "Process priority set to highest." << std::endl;
	constexpr int B = 8;
	constexpr int T = 1024;
	constexpr int C = 768;
	float* input = (float*)_mm_malloc(sizeof(float) * B * T * C, 64);
	float* output = (float*)_mm_malloc(sizeof(float) * B * T * C, 64);
	float* mean = (float*)_mm_malloc(sizeof(float) * B * T, 64);
	float* rstd = (float*)_mm_malloc(sizeof(float) * B * T, 64);
	float* weight = (float*)_mm_malloc(sizeof(float) * C, 64);
	float* bias = (float*)_mm_malloc(sizeof(float) * C, 64);
	double total = 0.0;
	constexpr int iters = 200;
	cout << "Input: " << input << endl << "Output: " << output << endl << "mean: " << mean << "rstd: " << rstd << "weight: " << weight << "bias: " << bias << endl;

    	// 创建随机数引擎
    	std::random_device rd;  // 用于获取随机数种子
        std::mt19937 gen(rd()); // 以随机设备生成的种子初始化Mersenne Twister生成器
        // 创建一个均匀分布的浮点数生成器
        std::uniform_real_distribution<> dis(1.0, 100.0); // 定义[1.0, 100.0)区间的均匀分布
	for (int i = 0; i < C; ++i) {
		weight[i] = dis(gen);
		bias[i] = dis(gen);
	}

	for (int i = 0; i < iters; ++i) {
		for (int j = 0; j < B * T * C; ++j) {
			input[j] = dis(gen);
		}
		auto start = std::chrono::high_resolution_clock::now();
		layernorm_forward_cpu<B, T, C>(output, mean, rstd, input, weight, bias);
		auto finish = std::chrono::high_resolution_clock::now();
		total += (finish - start).count();
	}
  	std::cout << "\nIters: " << iters << ", average elapsed time: " << total / 1000000000.0 / iters << " s\n";   // 输出耗时
  	return 0;
}
