/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_FULLY_CONNECTED_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_FULLY_CONNECTED_H_

#include <algorithm>
#include <cstdio>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "cfu.h"

namespace tflite {
namespace reference_integer_ops {

// For per-channel functions, since it is defined in quantization spec that
// weights are symmetric
// (https://www.tensorflow.org/lite/performance/quantization_spec#symmetric_vs_asymmetric),
// zero_point (params.weights_offset) is always 0.
// However, for per-tensor functions, params.weights_offset is still applied for
// backward compatibility.

#define MAX_CHANNEL 64
#define ACCUM_DEPTH 64

inline uint32_t pack_data(int8_t a, int8_t b, int8_t c, int8_t d) {
    return ((uint8_t)a << 24) | ((uint8_t)b << 16) | ((uint8_t)c << 8) | ((uint8_t)d);
}

inline void send_blockA(int8_t blockA[MAX_CHANNEL][ACCUM_DEPTH], int8_t M_depth, int8_t K_depth) {
    int depth = 0;
    for (int m = 0; m < M_depth; m += 4) {
        for (int k = 0; k < K_depth; k++) {
            const int32_t packed_data = pack_data(blockA[m][k], blockA[m + 1][k], blockA[m + 2][k], blockA[m + 3][k]);
            cfu_op0(1, packed_data, depth++);
        }
    }
}

inline void send_blockB(int8_t blockB[ACCUM_DEPTH][1], int8_t K_depth) {
    int depth = 0;
    for (int k = 0; k < K_depth; k++) {
        const int32_t packed_data = pack_data(blockB[k][0], 0, 0, 0);
        cfu_op0(2, packed_data, depth++);
    }
    
}

inline void receive_blockC(int32_t blockC[MAX_CHANNEL][1], int8_t M_depth) {
    int depth = 0;
    for (int m = 0; m < M_depth; m++) {
        blockC[m][0] = (int32_t)cfu_op0(3, 0, depth++);
    }
    
}

inline void FullyConnectedPerChannel(
    const FullyConnectedParams& params,
    const int32_t* output_multiplier,
    const int* output_shift,
    const RuntimeShape& input_shape,
    const int8_t* input_data,
    const RuntimeShape& filter_shape,
    const int8_t* filter_data,
    const RuntimeShape& bias_shape,
    const int32_t* bias_data,
    const RuntimeShape& output_shape,
    int8_t* output_data) {
    const int32_t input_offset = params.input_offset;
    const int32_t output_offset = params.output_offset;
    const int32_t output_activation_min = params.quantized_activation_min;
    const int32_t output_activation_max = params.quantized_activation_max;
    TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
    TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 2);

    TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
    const int filter_dim_count = filter_shape.DimensionsCount();
    const int batches = output_shape.Dims(0);
    const int output_depth = output_shape.Dims(1);
    TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
    const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
    for (int b = 0; b < batches; ++b) {
        for (int out_c = 0; out_c < output_depth; ++out_c) {
            int32_t acc = 0;
            for (int d = 0; d < accum_depth; ++d) {
                int32_t input_val = input_data[b * accum_depth + d];
                int32_t filter_val = filter_data[out_c * accum_depth + d];
                acc += filter_val * (input_val + input_offset);
            }

            if (bias_data) {
                acc += bias_data[out_c];
            }
            acc = MultiplyByQuantizedMultiplier(acc, output_multiplier[out_c],
                                                output_shift[out_c]);
            acc += output_offset;
            acc = std::max(acc, output_activation_min);
            acc = std::min(acc, output_activation_max);
            output_data[out_c + output_depth * b] = static_cast<int8_t>(acc);
        }
        
    }
}

template <typename AccumScalar>
inline void FullyConnectedPerChannel(
    const FullyConnectedParams& params,
    const int32_t* output_multiplier,
    const int* output_shift,
    const RuntimeShape& input_shape,
    const int16_t* input_data,
    const RuntimeShape& filter_shape,
    const int8_t* filter_data,
    const RuntimeShape& bias_shape,
    const AccumScalar* bias_data,
    const RuntimeShape& output_shape,
    int16_t* output_data) {
    const int32_t output_activation_min = params.quantized_activation_min;
    const int32_t output_activation_max = params.quantized_activation_max;
    TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
    TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);

    TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
    const int filter_dim_count = filter_shape.DimensionsCount();
    const int output_dim_count = output_shape.DimensionsCount();
    const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
    const int output_depth = output_shape.Dims(output_dim_count - 1);
    TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
    const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
    for (int b = 0; b < batches; ++b) {
        for (int out_c = 0; out_c < output_depth; ++out_c) {
            AccumScalar acc = 0;
            for (int d = 0; d < accum_depth; ++d) {
                int32_t input_val = input_data[b * accum_depth + d];
                int32_t filter_val = filter_data[out_c * accum_depth + d];
                acc += filter_val * input_val;
            }
            if (bias_data) {
                acc += bias_data[out_c];
            }
            int32_t acc_scaled = MultiplyByQuantizedMultiplier(
                acc, output_multiplier[out_c], output_shift[out_c]);
            acc_scaled = std::max(acc_scaled, output_activation_min);
            acc_scaled = std::min(acc_scaled, output_activation_max);
            output_data[out_c + output_depth * b] = static_cast<int16_t>(acc_scaled);
        }
    }
}

inline void FullyConnected(
    const FullyConnectedParams& params,
    const RuntimeShape& input_shape,
    const int8_t* input_data,
    const RuntimeShape& filter_shape,
    const int8_t* filter_data,
    const RuntimeShape& bias_shape,
    const int32_t* bias_data,
    const RuntimeShape& output_shape,
    int8_t* output_data) {
    const int32_t input_offset = params.input_offset;
    // const int32_t filter_offset = params.weights_offset;
    const int32_t output_offset = params.output_offset;
    const int32_t output_multiplier = params.output_multiplier;
    const int output_shift = params.output_shift;
    const int32_t output_activation_min = params.quantized_activation_min;
    const int32_t output_activation_max = params.quantized_activation_max;

    // printf("filter_offset: %ld\n", filter_offset);

    TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
    TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);

    TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
    const int filter_dim_count = filter_shape.DimensionsCount();
    const int output_dim_count = output_shape.DimensionsCount();
    const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
    const int output_depth = output_shape.Dims(output_dim_count - 1);
    TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
    const int accum_depth = filter_shape.Dims(filter_dim_count - 1);

    // printf("output_depth: %d\n", output_depth);
    // printf("accum_depth: %d\n", accum_depth);

    for (int b = 0; b < batches; ++b) {
        
        int8_t weight_im2col[MAX_CHANNEL][ACCUM_DEPTH] = {0};
        int8_t input_im2col[ACCUM_DEPTH][1] = {0};
        int32_t result_im2col[MAX_CHANNEL][1] = {0}; 

        for (int d = 0; d < accum_depth; ++d){
            input_im2col[d][0] = input_data[b * accum_depth + d];
        }

        for (int out_c = 0; out_c < output_depth; ++out_c) {

            for (int d = 0; d < accum_depth; d += 4) {

                // int32_t input_val = input_data[b * accum_depth + d];
                // int32_t filter_val = filter_data[out_c * accum_depth + d];
                // acc += filter_val * (input_val + input_offset);
                uint32_t* filter_ptr = (uint32_t*)(filter_data + out_c * accum_depth + d);
                // filter_val = filter_data[out_c * accum_depth + d];
                uint32_t* weight_ptr = (uint32_t*)&weight_im2col[out_c][d];
                *weight_ptr = *filter_ptr;
            }
        }

        send_blockA(weight_im2col, output_depth, accum_depth);
        send_blockB(input_im2col, accum_depth);
        cfu_op0(0, pack_data(0, accum_depth, output_depth, 1), input_offset);
        cfu_op0(4, 0, 0);
        receive_blockC(result_im2col, output_depth);

        // for (int out_c = 0; out_c < output_depth; ++out_c){
        //     for (int d = 0; d < accum_depth; ++d){
        //         result_im2col[out_c][0] += filter_offset * (input_im2col[d][0] + input_offset);
        //     }
        // }

        for (int out_c = 0; out_c < output_depth; ++out_c){
            if (bias_data) {
                result_im2col[out_c][0] += bias_data[out_c];
            }
            result_im2col[out_c][0] = MultiplyByQuantizedMultiplier(result_im2col[out_c][0], output_multiplier,
                                                output_shift);
            result_im2col[out_c][0] += output_offset;
            result_im2col[out_c][0] = std::max(result_im2col[out_c][0], output_activation_min);
            result_im2col[out_c][0] = std::min(result_im2col[out_c][0], output_activation_max);
            output_data[out_c + output_depth * b] = static_cast<int8_t>(result_im2col[out_c][0]);
        }
    }
}

inline void FullyConnectedWithPackedInt4Weights(
    const FullyConnectedParams& params,
    const RuntimeShape& input_shape,
    const int8_t* input_data,
    const RuntimeShape& filter_shape,
    const int8_t* filter_data,
    int8_t* unpacked_filter_data,
    const RuntimeShape& bias_shape,
    const int32_t* bias_data,
    const RuntimeShape& output_shape,
    int8_t* output_data) {
    TFLITE_DCHECK_NE(unpacked_filter_data, nullptr);
    tflite::tensor_utils::UnpackDenseInt4IntoInt8(
        filter_data, filter_shape.FlatSize(), unpacked_filter_data);
    FullyConnected(params, input_shape, input_data, filter_shape,
                   unpacked_filter_data, bias_shape, bias_data, output_shape,
                   output_data);
}

template <typename AccumScalar>
inline void FullyConnected(
    const FullyConnectedParams& params,
    const RuntimeShape& input_shape,
    const int16_t* input_data,
    const RuntimeShape& filter_shape,
    const int8_t* filter_data,
    const RuntimeShape& bias_shape,
    const AccumScalar* bias_data,
    const RuntimeShape& output_shape,
    int16_t* output_data) {
    const int32_t filter_offset = params.weights_offset;
    const int32_t output_multiplier = params.output_multiplier;
    const int output_shift = params.output_shift;
    const int32_t output_activation_min = params.quantized_activation_min;
    const int32_t output_activation_max = params.quantized_activation_max;
    TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
    TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);

    TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
    const int filter_dim_count = filter_shape.DimensionsCount();
    const int output_dim_count = output_shape.DimensionsCount();
    const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
    const int output_depth = output_shape.Dims(output_dim_count - 1);
    TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
    const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
    for (int b = 0; b < batches; ++b) {
        for (int out_c = 0; out_c < output_depth; ++out_c) {
            AccumScalar acc = 0;
            for (int d = 0; d < accum_depth; ++d) {
                int32_t input_val = input_data[b * accum_depth + d];
                int32_t filter_val = filter_data[out_c * accum_depth + d];
                acc += (filter_val + filter_offset) * input_val;
            }
            if (bias_data) {
                acc += bias_data[out_c];
            }
            int32_t acc_scaled =
                MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
            acc_scaled = std::max(acc_scaled, output_activation_min);
            acc_scaled = std::min(acc_scaled, output_activation_max);
            output_data[out_c + output_depth * b] = static_cast<int16_t>(acc_scaled);
        }
    }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_FULLY_CONNECTED_H_