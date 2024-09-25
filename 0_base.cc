#include <iostream>
#include <random>
#include <chrono>
#include <emmintrin.h>
#include <immintrin.h>

// NOTE: 基准实现copy自https://zhuanlan.zhihu.com/p/694974164
// GPT-2 layernorm forward pass
void layernorm_forward_cpu(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C) {
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            const float* x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalized output
                float o = n * weight[i] + bias[i]; // scale and shift it
                out_bt[i] = o; // write
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
	float* input = new float[B * T * C];
	float* output = new float[B * T * C];
	float* mean = new float[B * T];
	float* rstd = new float[B * T];
	float* weight = new float[C];
	float* bias = new float[C];
	double total = 0.0;
	constexpr int iters = 200;
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
		layernorm_forward_cpu(output, mean, rstd, input, weight, bias, B, T, C);
		auto finish = std::chrono::high_resolution_clock::now();
		total += (finish - start).count();
	}
  	std::cout << "\nIters: " << iters << ", average elapsed time: " << total / 1000000000.0 / iters << " s\n";   // 输出耗时
  	return 0;
}
