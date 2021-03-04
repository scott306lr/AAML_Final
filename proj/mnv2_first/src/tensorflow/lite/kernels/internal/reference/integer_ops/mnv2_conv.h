/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
   Copyright 2021 The CFU PLayground Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_MNV2_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_MNV2_CONV_H_

#include "tensorflow/lite/kernels/internal/common.h"
#include "tf_util/print_params.h"

//
// This file contains specialized conv 2D implementations to support
// MobileNet v2 models
//
namespace tflite {
namespace reference_integer_ops {

// Fixed-point per-channel-quantization convolution reference kernel.
inline void Mnv2ConvPerChannel1x1(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
#ifdef SHOW_CONV_PARAMS
  print_conv_params(params, input_shape, filter_shape, output_shape);
#endif

  // Get parameters.
  const int32_t input_offset = params.input_offset;  // r = s(q - Z)
  const int32_t output_offset = params.output_offset;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int y = 0; y < output_height; ++y) {
    for (int x = 0; x < output_width; ++x) {
      for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
        int32_t acc = 0;

        for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
          int32_t input_val =
              input_data[Offset(input_shape, 0, y, x, in_channel)];
          int32_t filter_val =
              filter_data[Offset(filter_shape, out_channel, 0, 0, in_channel)];
          // Accumulate with 32 bits accumulator.
          // In the nudging process during model quantization, we force
          // real value of 0.0 be represented by a quantized value. This
          // guarantees that the input_offset is a int8_t, even though
          // it is represented using int32_t. int32_t += int8_t *
          // (int8_t - int8_t) so the highest value we can get from each
          // accumulation is [-127, 127] * ([-128, 127] -
          // [-128, 127]), which is [-32512, 32512]. log2(32512)
          // = 14.98, which means we can accumulate at least 2^16
          // multiplications without overflow. The accumulator is
          // applied to a filter so the accumulation logic will hold as
          // long as the filter size (filter_y * filter_x * in_channel)
          // does not exceed 2^16, which is the case in all the models
          // we have seen so far.
          // TODO(jianlijianli): Add a check to make sure the
          // accumulator depth is smaller than 2^16.
          acc += filter_val * (input_val + input_offset);
        }

        if (bias_data) {
          acc += bias_data[out_channel];
        }
        acc = MultiplyByQuantizedMultiplier(acc, output_multiplier[out_channel],
                                            output_shift[out_channel]);
        acc += output_offset;
        acc = std::max(acc, output_activation_min);
        acc = std::min(acc, output_activation_max);
        output_data[Offset(output_shape, 0, y, x, out_channel)] =
            static_cast<int8_t>(acc);
      }
    }
  }
}
}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_MNV2_CONV_H_
