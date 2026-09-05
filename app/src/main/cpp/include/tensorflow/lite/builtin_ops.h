#ifndef TENSORFLOW_LITE_BUILTIN_OPS_H_
#define TENSORFLOW_LITE_BUILTIN_OPS_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  kTfLiteBuiltinAdd = 0,
  kTfLiteBuiltinAveragePool2d = 1,
  kTfLiteBuiltinConcatenation = 2,
  kTfLiteBuiltinConv2d = 3,
  kTfLiteBuiltinDepthwiseConv2d = 4,
  kTfLiteBuiltinFullyConnected = 9,
  kTfLiteBuiltinReshape = 22,
  kTfLiteBuiltinResizeBilinear = 23,
  kTfLiteBuiltinSoftmax = 25,
  kTfLiteBuiltinTranspose = 39,
  kTfLiteBuiltinPad = 52,
  kTfLiteBuiltinStridedSlice = 63,
  kTfLiteBuiltinDiv = 64,
  kTfLiteBuiltinSub = 65,
  kTfLiteBuiltinMul = 74,
  kTfLiteBuiltinCustom = 300,
} TfLiteBuiltinOperator;

#ifdef __cplusplus
}
#endif

#endif  // TENSORFLOW_LITE_BUILTIN_OPS_H_
