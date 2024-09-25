#include <iostream>
#include <random>
#include <chrono>
#include <emmintrin.h>
#include <immintrin.h>

using namespace std;

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
	    __m256 va = _mm256_loadu_ps(x);
	    for (int i = 1; i < C; i += 8) {
		__m256 vb = _mm256_loadu_ps(x + i);
		va = _mm256_add_ps(va, vb);
	    }
	    __m256 vb = _mm256_hadd_ps(va, va);
	    __m256 vc = _mm256_hadd_ps(vb, vb);
	    __m128 lo = _mm256_extractf128_ps(vc, 0);
	    __m128 hi = _mm256_extractf128_ps(vc, 1);
	    float m = _mm_cvtss_f32(lo) + _mm_cvtss_f32(hi);
            m = m / C;
            // calculate the variance (without any bias correction)
	    __m256 square_sum = _mm256_setzero_ps();
	    __m256 broadcast_mean = _mm256_set1_ps(m);
            for (int i = 0; i < C; i += 8) {
		__m256 va = _mm256_loadu_ps(x + i);
		__m256 vb = _mm256_sub_ps(va, broadcast_mean);
		__m256 vc = _mm256_mul_ps(vb, vb);
		square_sum = _mm256_add_ps(square_sum, vc);
            }
	    __m256 vd = _mm256_hadd_ps(square_sum, square_sum);
	    __m256 ve = _mm256_hadd_ps(vd, vd);
	    lo = _mm256_extractf128_ps(ve, 0);
	    hi = _mm256_extractf128_ps(ve, 1);
	    float v = (_mm_cvtss_f32(lo) + _mm_cvtss_f32(hi)) / C;
            // calculate the rstd
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
	    __m256 broadcast_rstd = _mm256_set1_ps(s);
            for (int i = 0; i < C; i += 8) {
		__m256 va = _mm256_loadu_ps(x + i);
		__m256 vb = _mm256_mul_ps(_mm256_sub_ps(va, broadcast_mean), broadcast_rstd);
		__m256 vw = _mm256_loadu_ps(weight + i);
		__m256 vbias = _mm256_loadu_ps(bias + i);
		__m256 vv = _mm256_add_ps(_mm256_mul_ps(vb, vw), vbias);
		_mm256_storeu_ps(out_bt + i, vv);
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

int main() {
	constexpr int B = 8;
	constexpr int T = 1024;
	constexpr int C = 768;
	float* input = (float*)_mm_malloc(sizeof(float) * B * T * C, 32);
	float* output = (float*)_mm_malloc(sizeof(float) * B * T * C, 32);
	float* mean = (float*)_mm_malloc(sizeof(float) * B * T, 32);
	float* rstd = (float*)_mm_malloc(sizeof(float) * B * T, 32);
	float* weight = (float*)_mm_malloc(sizeof(float) * C, 32);
	float* bias = (float*)_mm_malloc(sizeof(float) * C, 32);
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
